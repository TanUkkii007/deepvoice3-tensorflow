import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited, assume, HealthCheck
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.deepvoice3 import Decoder, MultiHopAttentionArgs, DecoderPreNetCNNArgs
import tensorflow.contrib.eager as tfe

tfe.enable_eager_execution()

even_number = lambda x: x % 2 == 0


@composite
def query_tensor(draw, batch_size, c, t_size=integers(3, 8), elements=integers(-1, 1)):
    t = draw(t_size)
    btc = draw(arrays(dtype=np.float32, shape=[batch_size, t, c], elements=elements))
    return btc


@composite
def memory_tensor(draw, batch_size, input_length=integers(5, 20),
                  embed_dim=integers(4, 20).filter(even_number), elements=integers(-1, 1)):
    il = draw(input_length)
    md = draw(embed_dim)
    t = draw(arrays(dtype=np.float32, shape=[batch_size, il, md], elements=elements))
    return t


@composite
def mha_arg(draw, out_channels, kernel_size=integers(2, 10), dilation=integers(1, 20)):
    ks = draw(kernel_size)
    dl = draw(dilation)
    return MultiHopAttentionArgs(out_channels, ks, dl, dropout=0.0)


@composite
def all_args(draw, batch_size=integers(1, 3), query_channels=integers(2, 20).filter(even_number),
             in_dim=integers(2, 20), r=integers(1, 1)):
    bs = draw(batch_size)
    _in_dim = draw(in_dim)
    _r = draw(r)
    qc = draw(query_channels)
    query = draw(query_tensor(bs, _in_dim * _r))
    mha = draw(mha_arg(qc))
    memory = draw(memory_tensor(bs))
    return query, mha, memory, _in_dim, _r


class DecoderTest(tf.test.TestCase):

    @given(args=all_args(), num_preattention=integers(1, 3), preattention_kernel_size=integers(1, 9),
           num_mha=integers(1, 4))
    @settings(max_examples=10, timeout=unlimited, suppress_health_check=[HealthCheck.too_slow])
    def test_decoder(self, args, num_preattention, preattention_kernel_size, num_mha):
        tf.set_random_seed(12345678)
        query, mha_arg, memory, in_dim, r = args
        batch_size = 1
        max_positions = 30
        T_query = query.shape[1]
        embed_dim = memory.shape[2]
        T_memory = memory.shape[1]
        assume(T_query < max_positions and T_memory < max_positions)

        print("query", query)
        print("memory", memory)

        preattention_args = [DecoderPreNetCNNArgs(mha_arg.out_channels, preattention_kernel_size, dilation=2 ** i) for i
                             in range(num_preattention)]

        def one_tenth_initializer(length):
            half = length // 2
            return np.stack([0.1 * -1 * np.ones(half), 0.1 * np.ones(half)]).reshape(length, order='F')

        # prenet_weight_initializer = tf.constant_initializer(one_tenth_initializer(preattention_in_features * mha_arg.out_channels))
        prenet_weight_initializer = tf.ones_initializer()
        attention_key_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(embed_dim * embed_dim))
        attention_value_projection_weight_initializer = tf.constant_initializer(
            one_tenth_initializer(embed_dim * embed_dim))
        attention_kernel_initializer = tf.constant_initializer(
            one_tenth_initializer(mha_arg.kernel_size * mha_arg.out_channels * mha_arg.out_channels * 2))
        _attention_weight = one_tenth_initializer(mha_arg.out_channels * embed_dim)
        attention_query_projection_weight_initializer = tf.constant_initializer(_attention_weight)
        attention_out_projection_weight_initializer = tf.constant_initializer(_attention_weight)
        last_conv_kernel_initializer = tf.constant_initializer(one_tenth_initializer(mha_arg.out_channels * in_dim * r))
        done_weight_initializer = tf.ones_initializer()

        decoder = Decoder(embed_dim, in_dim, r, max_positions, preattention=preattention_args,
                          mh_attentions=(mha_arg,) * num_mha,
                          dropout=0.0, is_incremental=False,
                          prenet_weight_initializer=prenet_weight_initializer,
                          attention_key_projection_weight_initializer=attention_key_projection_weight_initializer,
                          attention_value_projection_weight_initializer=attention_value_projection_weight_initializer,
                          attention_kernel_initializer=attention_kernel_initializer,
                          attention_query_projection_weight_initializer=attention_query_projection_weight_initializer,
                          attention_out_projection_weight_initializer=attention_out_projection_weight_initializer,
                          last_conv_kernel_initializer=last_conv_kernel_initializer,
                          done_weight_initializer=done_weight_initializer,
                          )

        decoder_online = Decoder(embed_dim, in_dim, r, max_positions, preattention=preattention_args,
                                 mh_attentions=(mha_arg,) * num_mha,
                                 dropout=0.0, max_decoder_steps=T_query, min_decoder_steps=T_query, is_incremental=True,
                                 prenet_weight_initializer=prenet_weight_initializer,
                                 attention_key_projection_weight_initializer=attention_key_projection_weight_initializer,
                                 attention_value_projection_weight_initializer=attention_value_projection_weight_initializer,
                                 attention_kernel_initializer=attention_kernel_initializer,
                                 attention_query_projection_weight_initializer=attention_query_projection_weight_initializer,
                                 attention_out_projection_weight_initializer=attention_out_projection_weight_initializer,
                                 last_conv_kernel_initializer=last_conv_kernel_initializer,
                                 done_weight_initializer=done_weight_initializer,
                                 )

        # frame pos start with 1
        frame_positions = tf.zeros(shape=(batch_size, T_query), dtype=tf.int32) + tf.range(1, T_query + 1,
                                                                                           dtype=tf.int32)
        text_positions = tf.zeros(shape=(batch_size, T_memory), dtype=tf.int32) + tf.range(0, T_memory, dtype=tf.int32)

        keys, values = tf.constant(memory), tf.constant(memory)
        out, done = decoder((keys, values), input=tf.constant(query),
                            frame_positions=frame_positions, text_positions=text_positions)

        out_online = decoder_online((keys, values),
                                    frame_positions=frame_positions, text_positions=text_positions,
                                    test_inputs=tf.constant(query))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     out = sess.run(out)
        #     out_online = sess.run(out_online)

        print(out)
        print(out_online)
        print("-" * 100)
        self.assertAllClose(out, out_online)

    @given(args=all_args(), num_preattention=integers(1, 3), preattention_kernel_size=integers(1, 9),
           num_mha=integers(1, 4))
    @settings(max_examples=10, timeout=unlimited)
    def test_decoder_inference(self, args, num_preattention, preattention_kernel_size, num_mha):
        tf.set_random_seed(12345678)
        query, mha_arg, memory, in_dim, r = args
        batch_size = 1
        max_positions = 30
        T_query = query.shape[1]
        embed_dim = memory.shape[2]
        T_memory = memory.shape[1]
        assume(T_query < max_positions and T_memory < max_positions)
        print("query", query)
        print("memory", memory)

        preattention_args = [DecoderPreNetCNNArgs(mha_arg.out_channels, preattention_kernel_size, dilation=2 ** i) for i
                             in range(num_preattention)]
        decoder_online = Decoder(embed_dim, in_dim, r, max_positions, preattention=preattention_args,
                                 mh_attentions=(mha_arg,) * num_mha,
                                 dropout=0.0, max_decoder_steps=T_query, min_decoder_steps=T_query, is_incremental=True)

        # frame pos start with 1
        frame_positions = tf.zeros(shape=(batch_size, T_query), dtype=tf.int32) + tf.range(1, T_query + 1,
                                                                                           dtype=tf.int32)
        text_positions = tf.zeros(shape=(batch_size, T_memory), dtype=tf.int32) + tf.range(0, T_memory, dtype=tf.int32)

        keys, values = tf.constant(memory), tf.constant(memory)
        out_online = decoder_online((keys, values),
                                    frame_positions=frame_positions, text_positions=text_positions)

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     out_online = sess.run(out_online)

        print(out_online)
        print("-" * 100)
