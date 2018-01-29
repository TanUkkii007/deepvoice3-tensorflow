import tensorflow as tf
import numpy as np
from hypothesis import given, assume, settings, unlimited
from hypothesis.strategies import integers
from deepvoice3_tensorflow.ops import causal_conv, Conv1dIncremental


class Conv1dIncrementalTest(tf.test.TestCase):

    @given(n=integers(5, 20), dilation=integers(1, 20))
    def test_causal_conv(self, n, dilation):
        assume(n - dilation - 1 > 0)
        filter = [1, 1]
        x1 = np.arange(1, n, dtype=np.float32)
        x = np.append(x1, x1)
        x = np.reshape(x, [2, n - 1, 1])
        f = np.reshape(np.array(filter, dtype=np.float32), [2, 1, 1])
        out = causal_conv(x, f, dilation)

        with self.test_session() as sess:
            result = sess.run(out)

        # Causal convolution using numpy
        f = [filter[0]] + [0] * (dilation - 1) + [filter[1]]
        ref = np.convolve(x1, f, mode='valid')
        ref = np.append(ref, ref)
        ref = np.reshape(ref, [2, n - dilation - 1, 1])

        self.assertAllEqual(ref, result)

    @given(B=integers(1, 3), T=integers(10, 30), C=integers(1, 4), kernel_size=integers(2, 9), dilation=integers(1, 27))
    @settings(max_examples=10, timeout=unlimited)
    def test_conv1d_incremental(self, kernel_size, dilation, T, B, C):

        bct_value = np.zeros(shape=[B, C, T], dtype=np.float32) + (
                np.zeros(shape=[C, T], dtype=np.float32) + np.arange(0, T, dtype=np.float32))
        btc_value = bct_value.transpose((0, 2, 1))
        # padding to ensure no time shift
        padding = (kernel_size - 1) * dilation
        btc_value_pad = np.pad(btc_value, [[0, 0], [padding, 0], [0, 0]], 'constant')
        filter_value = np.ones(shape=[C * 2, C, kernel_size], dtype=np.float32)
        out = causal_conv(btc_value_pad, filter_value.transpose([2, 1, 0]), dilation)
        with self.test_session() as sess:
            output_causal_conv = sess.run(out)
            print(output_causal_conv)


        btc_one = tf.placeholder(dtype=tf.float32, shape=[B, 1, C])
        filter = tf.Variable(initial_value=tf.ones(shape=[C * 2, C, kernel_size]))
        conv1d_incremental = Conv1dIncremental(filter, C, C * 2, kernel_size=kernel_size, dilation=dilation)
        buffer_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        input_buffer_pf = tf.placeholder(dtype=tf.float32, shape=[B, buffer_size, C])
        output_conv, next_input_buffer = conv1d_incremental.apply(btc_one, training=False,                                                                  input_buffer=input_buffer_pf)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_conv_online = []
            input_buffer = np.zeros(shape=[B, buffer_size, C], dtype=np.float32)

            for t in range(T):
                result, input_buffer = sess.run([output_conv, next_input_buffer], feed_dict={
                    btc_one: btc_value[:, t, :].reshape(B, -1, C),
                    input_buffer_pf: input_buffer
                })
                output_conv_online += [result]

        print(len(output_conv_online))

        output_conv_online = np.stack(output_conv_online).squeeze(axis=2)
        output_conv_online = output_conv_online.transpose((1, 0, 2))

        print(output_conv_online)

        self.assertAllEqual(output_causal_conv, output_conv_online)
