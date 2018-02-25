import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers, floats, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.deepvoice3 import ScaledDotProductAttentionMechanism, AttentionLayer, CNNAttentionWrapper, MultiHopAttention, MultiHopAttentionArgs
from tensorflow.python.util import nest


@composite
def attention_tensors(draw, b_size=integers(1, 5), t_query_size=integers(2, 20), c_size=integers(1, 10),
                      t_encoder_size=integers(2, 10), embed_dim=integers(1, 10), elements=integers(-5, 5)):
    b = draw(b_size)
    t_query = draw(t_query_size)
    c = draw(c_size)
    t_encoder = draw(t_encoder_size)
    embed = draw(embed_dim)
    query = draw(arrays(dtype=np.float32, shape=[b, t_query, c], elements=elements))
    encoder = draw(arrays(dtype=np.float32, shape=[b, t_encoder, embed], elements=elements))
    return (b, t_query, c, t_encoder, embed, query, encoder)


class AttentionLayerTest(tf.test.TestCase):

    @given(tensors=attention_tensors(), dropout=floats(0.5, 1.0, allow_nan=False))
    @settings(max_examples=10, timeout=unlimited)
    def test_attention(self, tensors, dropout):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        query = tf.constant(query)
        encoder_out = tf.constant(encoder_out)

        attention_mechanism = ScaledDotProductAttentionMechanism(encoder_out, embed_dim)

        attention = AttentionLayer(attention_mechanism, C, dropout)
        output = attention.apply(query)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(output)

    @given(tensors=attention_tensors(), kernel_size=integers(2, 10),
           dilation=integers(1, 20), r=integers(1, 1))
    @settings(max_examples=10, timeout=unlimited)
    def test_cnn_attention_wrapper(self, tensors, kernel_size, dilation, r):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(C % 2 == 0)
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        dropout = 1.0
        out_channels = C
        query = tf.constant(query)
        encoder_out = tf.constant(encoder_out)

        attention_mechanism = ScaledDotProductAttentionMechanism(encoder_out, embed_dim)
        attention = CNNAttentionWrapper(attention_mechanism, C, out_channels, kernel_size, dilation, dropout,
                                        is_incremental=False, r=r, kernel_initializer_seed=123,
                                        weight_initializer_seed=456)
        output, _states = attention.apply(query, attention.zero_state(B, tf.float32))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(output)
        print(output)

        attention_incremental = CNNAttentionWrapper(attention_mechanism, C, out_channels, kernel_size, dilation,
                                                    dropout,
                                                    is_incremental=True, r=r, kernel_initializer_seed=123,
                                                    weight_initializer_seed=456)

        def condition(time, unused_inputs, unused_state, unused_outputs):
            return tf.less(time, T_query)

        def body(time, inputs, state, outputs):
            # ToDo: slice time:time+reduction_factor
            btc_one = tf.reshape(inputs[:, time:time + r, :], shape=(B, -1, C))
            out_online, next_states = attention_incremental.apply(btc_one, state)
            return (time + r, inputs, next_states, outputs.write(time, out_online))

        time = tf.constant(0)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T_query, element_shape=tf.TensorShape([B, 1, C]))
        zero_state = attention_incremental.zero_state(B, tf.float32)
        _, _, final_state, out_online_ta = tf.while_loop(condition, body, (time, query, zero_state, outputs_ta))
        out_online = nest.map_structure(lambda ta: ta.stack(), out_online_ta)

        output_online = tf.squeeze(out_online, axis=2)
        output_online = tf.transpose(output_online, perm=(1, 0, 2))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_online, alignments = sess.run([output_online, final_state.alignments])

        print(output_online)
        print("---------------")
        self.assertAllClose(output, output_online, atol=1e-4)

    @given(tensors=attention_tensors(b_size=integers(1, 2)), kernel_size=integers(2, 10),
           dilation=integers(1, 20), r=integers(1, 1))
    @settings(max_examples=10, timeout=unlimited)
    def test_multi_hop_attention(self, tensors, kernel_size, dilation, r):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(C % 2 == 0)
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        dropout = 1.0
        out_channels = C
        query = tf.constant(query)
        encoder_out = tf.constant(encoder_out)

        attention_mechanism = ScaledDotProductAttentionMechanism(encoder_out, embed_dim)
        args = [MultiHopAttentionArgs(out_channels, kernel_size, dilation, dropout, r, kernel_initializer_seed=123,
                                        weight_initializer_seed=456)] * 3
        attention = MultiHopAttention(attention_mechanism, C, args, is_incremental=False)
        output, _states = attention.apply(query, attention.zero_state(B, tf.float32))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output = sess.run(output)
        print(output)

        attention_incremental = MultiHopAttention(attention_mechanism, C, args, is_incremental=True)

        def condition(time, unused_inputs, unused_state, unused_outputs):
            return tf.less(time, T_query)

        def body(time, inputs, state, outputs):
            # ToDo: slice time:time+reduction_factor
            btc_one = tf.reshape(inputs[:, time:time + r, :], shape=(B, -1, C))
            out_online, next_states = attention_incremental.apply(btc_one, state)
            return (time + r, inputs, next_states, outputs.write(time, out_online))

        time = tf.constant(0)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T_query, element_shape=tf.TensorShape([B, 1, C]))
        zero_state = attention_incremental.zero_state(B, tf.float32)
        _, _, final_state, out_online_ta = tf.while_loop(condition, body, (time, query, zero_state, outputs_ta))
        out_online = nest.map_structure(lambda ta: ta.stack(), out_online_ta)

        output_online = tf.squeeze(out_online, axis=2)
        output_online = tf.transpose(output_online, perm=(1, 0, 2))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_online, alignments = sess.run([output_online, MultiHopAttention.average_alignment(final_state)])

        print(output_online)
        print("---------------")
        self.assertAllClose(output, output_online, atol=1e-4)

if __name__ == '__main__':
    tf.test.main()
