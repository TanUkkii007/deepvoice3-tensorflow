import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers, floats, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.deepvoice3 import ScaledDotProductAttentionMechanism, AttentionLayer, CNNAttentionWrapper, \
    MultiHopAttention, MultiHopAttentionArgs, CNNAttentionWrapperInput
from tensorflow.python.util import nest

@composite
def attention_tensors(draw, b_size=integers(1, 5), t_query_size=integers(2, 20), c_size=integers(1, 10),
                      t_encoder_size=integers(2, 10), embed_dim=integers(1, 10).filter(lambda x: x % 2 == 0),
                      query_elements=integers(-5, 5),
                      encoder_elements=integers(-5, 5)):
    b = draw(b_size)
    t_query = draw(t_query_size)
    c = draw(c_size)
    t_encoder = draw(t_encoder_size)
    embed = draw(embed_dim)
    query = draw(arrays(dtype=np.float32, shape=[b, t_query, c], elements=query_elements))
    encoder = draw(arrays(dtype=np.float32, shape=[b, t_encoder, embed], elements=encoder_elements))
    return (b, t_query, c, t_encoder, embed, query, encoder)


class AttentionLayerTest(tf.test.TestCase):

    @given(tensors=attention_tensors(), dropout=floats(0.0, 0.5, allow_nan=False))
    @settings(max_examples=10, timeout=unlimited)
    def test_attention(self, tensors, dropout):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        query = tf.constant(query)
        keys, values = tf.constant(encoder_out), tf.constant(encoder_out)

        attention_mechanism = ScaledDotProductAttentionMechanism(keys, values, embed_dim)

        attention = AttentionLayer(attention_mechanism, C, dropout)
        output = attention.apply(query)

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(output)

    @given(tensors=attention_tensors(), kernel_size=integers(2, 10),
           dilation=integers(1, 20), r=integers(1, 1))
    @settings(max_examples=10, timeout=unlimited)
    def test_cnn_attention_wrapper(self, tensors, kernel_size, dilation, r):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(C % 2 == 0)
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        dropout = 0.0
        out_channels = C
        query = tf.constant(query)
        keys, values = tf.constant(encoder_out), tf.constant(encoder_out)

        def one_tenth_initializer(length):
            half = length // 2
            return np.stack([0.1 * -1 * np.ones(half), 0.1 * np.ones(half)]).reshape(length, order='F')

        attention_key_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(embed_dim * embed_dim))
        attention_value_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(embed_dim * embed_dim))
        attention_kernel_initializer = tf.constant_initializer(
            one_tenth_initializer(kernel_size * out_channels * out_channels * 2))
        attention_query_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(out_channels * embed_dim))
        attention_out_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(out_channels * embed_dim))

        frame_pos_embed = tf.zeros(shape=(B, T_query, out_channels), dtype=tf.float32)

        attention_mechanism = ScaledDotProductAttentionMechanism(keys, values, embed_dim,
                                                                 key_projection_weight_initializer=attention_key_projection_weight_initializer,
                                                                 value_projection_weight_initializer=attention_value_projection_weight_initializer)
        attention = CNNAttentionWrapper(attention_mechanism, C, out_channels, kernel_size, dilation, dropout,
                                        is_incremental=False, r=r,
                                        kernel_initializer=attention_kernel_initializer,
                                        query_projection_weight_initializer=attention_query_projection_weight_initializer,
                                        out_projection_weight_initializer=attention_out_projection_weight_initializer)
        (output, _), _states = attention.apply(CNNAttentionWrapperInput(query, frame_pos_embed),
                                          attention.zero_state(B, tf.float32))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     output = sess.run(output)
        print(output)

        attention_incremental = CNNAttentionWrapper(attention_mechanism, C, out_channels, kernel_size, dilation,
                                                    dropout,
                                                    is_incremental=True, r=r,
                                                    kernel_initializer=attention_kernel_initializer,
                                                    query_projection_weight_initializer=attention_query_projection_weight_initializer,
                                                    out_projection_weight_initializer=attention_out_projection_weight_initializer)

        frame_pos_embed_incremental = tf.zeros(shape=(B, 1, out_channels), dtype=tf.float32)

        def condition(time, unused_inputs, unused_state, unused_outputs):
            return tf.less(time, T_query)

        def body(time, inputs, state, outputs):
            btc_one = tf.reshape(inputs[:, time:time + 1, :], shape=(B, -1, C))
            input = CNNAttentionWrapperInput(btc_one, frame_pos_embed_incremental)
            out_online, next_states = attention_incremental.apply(input, state)
            return (time + 1, inputs, next_states, outputs.write(time, out_online.query))

        time = tf.constant(0)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T_query, element_shape=tf.TensorShape([B, 1, C]))
        zero_state = attention_incremental.zero_state(B, tf.float32)
        _, _, final_state, out_online_ta = tf.while_loop(condition, body, (time, query, zero_state, outputs_ta))
        out_online = nest.map_structure(lambda ta: ta.stack(), out_online_ta)

        output_online = tf.squeeze(out_online, axis=2)
        output_online = tf.transpose(output_online, perm=(1, 0, 2))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     output_online, alignments = sess.run([output_online, final_state.alignments])

        print(output_online)
        print("---------------")
        self.assertAllClose(output, output_online)

    @given(tensors=attention_tensors(b_size=integers(1, 2)), kernel_size=integers(2, 10),
           dilation=integers(1, 20), r=integers(1, 1))
    @settings(max_examples=10, timeout=unlimited)
    def test_multi_hop_attention(self, tensors, kernel_size, dilation, r):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(C % 2 == 0)
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        dropout = 0.0
        out_channels = C
        query = tf.constant(query)
        keys, values = tf.constant(encoder_out), tf.constant(encoder_out)
        frame_pos_embed = tf.zeros(shape=(B, T_query, out_channels), dtype=tf.float32)

        def one_tenth_initializer(length):
            half = length // 2
            return np.stack([0.1 * -1 * np.ones(half), 0.1 * np.ones(half)]).reshape(length, order='F')

        attention_key_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(embed_dim * embed_dim))
        attention_value_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(embed_dim * embed_dim))
        attention_kernel_initializer = tf.constant_initializer(
            one_tenth_initializer(kernel_size * out_channels * out_channels * 2))
        attention_query_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(out_channels * embed_dim))
        attention_out_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(out_channels * embed_dim))

        attention_mechanism = ScaledDotProductAttentionMechanism(keys, values, embed_dim,
                                                                 key_projection_weight_initializer=attention_key_projection_weight_initializer,
                                                                 value_projection_weight_initializer=attention_value_projection_weight_initializer)
        args = [MultiHopAttentionArgs(out_channels, kernel_size, dilation, dropout)] * 3
        attention = MultiHopAttention(attention_mechanism, C, args, r, is_incremental=False,
                                      kernel_initializer=attention_kernel_initializer,
                                      query_projection_weight_initializer=attention_query_projection_weight_initializer,
                                      out_projection_weight_initializer=attention_out_projection_weight_initializer)
        input = CNNAttentionWrapperInput(query, frame_pos_embed)
        (output, _), _states = attention.apply(input, attention.zero_state(B, tf.float32))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     output = sess.run(output)
        print(output)

        attention_incremental = MultiHopAttention(attention_mechanism, C, args, r, is_incremental=True,
                                                  kernel_initializer=attention_kernel_initializer,
                                                  query_projection_weight_initializer=attention_query_projection_weight_initializer,
                                                  out_projection_weight_initializer=attention_out_projection_weight_initializer)

        frame_pos_embed_incremental = tf.zeros(shape=(B, 1, out_channels), dtype=tf.float32)

        def condition(time, unused_inputs, unused_state, unused_outputs):
            return tf.less(time, T_query)

        def body(time, inputs, state, outputs):
            # ToDo: slice time:time+reduction_factor
            btc_one = tf.reshape(inputs[:, time:time + r, :], shape=(B, -1, C))
            input = CNNAttentionWrapperInput(btc_one, frame_pos_embed_incremental)
            out_online, next_states = attention_incremental.apply(input, state)
            return (time + r, inputs, next_states, outputs.write(time, out_online.query))

        time = tf.constant(0)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T_query, element_shape=tf.TensorShape([B, 1, C]))
        zero_state = attention_incremental.zero_state(B, tf.float32)
        _, _, final_state, out_online_ta = tf.while_loop(condition, body, (time, query, zero_state, outputs_ta))
        out_online = nest.map_structure(lambda ta: ta.stack(), out_online_ta)

        output_online = tf.squeeze(out_online, axis=2)
        output_online = tf.transpose(output_online, perm=(1, 0, 2))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     output_online, alignments = sess.run([output_online, MultiHopAttention.average_alignment(final_state)])

        print(output_online)
        print("---------------")
        self.assertAllClose(output, output_online)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()
