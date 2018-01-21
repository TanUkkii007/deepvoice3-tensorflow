import tensorflow as tf
import numpy as np
from deepvoice3_tensorflow.ops import causal_conv, Conv1dIncremental


class Conv1dIncrementalTest(tf.test.TestCase):

    def __test_causal_conv(self, n, dilation):
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

    def test_causal_conv(self):
        for n in [5, 11, 21]:
            for dilation in [1, 2, 3, 4, 5, 6]:
                if n - dilation - 1 > 0:
                    with self.subTest(n=n, dilation=dilation):
                        self.__test_causal_conv(n, dilation)

    def __test_conv1d_incremental(self, kernel_size, dilation, T, B, C):

        bct_value = np.zeros(shape=[B, C, T], dtype=np.float32) + (
                np.zeros(shape=[C, T], dtype=np.float32) + np.arange(0, T, dtype=np.float32))
        btc_value = bct_value.transpose((0, 2, 1))
        # padding to ensure no time shift
        padding = (kernel_size - 1) * dilation
        btc_value = np.pad(btc_value, [[0, 0], [padding, 0], [0, 0]], 'constant')
        filter_value = np.ones(shape=[C * 2, C, kernel_size], dtype=np.float32)
        out = causal_conv(btc_value, filter_value.transpose([2, 1, 0]), dilation)
        with self.test_session() as sess:
            output_causal_conv = sess.run(out)
            print(output_causal_conv)

        bct = tf.Variable(initial_value=tf.zeros([B, C, T]) + tf.range(0, T, dtype=tf.float32), trainable=False)
        # B, T, C
        btc = tf.transpose(bct, perm=[0, 2, 1])
        filter = tf.Variable(initial_value=tf.ones(shape=[C * 2, C, kernel_size]))
        conv1d_incremental = Conv1dIncremental(filter, C, C * 2, kernel_size=kernel_size, dilation=dilation)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_conv_online = []
            input_buffer = None
            for t in range(T):
                inputs = tf.reshape(btc[:, t, :], shape=(B, -1, C))
                output_conv = conv1d_incremental.apply(inputs, training=False, input_buffer=input_buffer)
                result, input_buffer = sess.run(output_conv)
                input_buffer = tf.constant(input_buffer)
                output_conv_online += [result]

        print(len(output_conv_online))

        output_conv_online = np.stack(output_conv_online).squeeze(axis=2)
        output_conv_online = output_conv_online.transpose((1, 0, 2))

        print(output_conv_online)

        # ToDo: causal_conv does not provide full time series. Its padding is not enough.
        self.assertAllEqual(output_causal_conv, output_conv_online)

    def test_conv1d_incremental(self):
        for B in [1]:
            for T in [10]:
                for C in [1, 2, 4]:
                    for kernel_size in [2, 3, 4, 5, 6, 7, 8, 9]:
                        for dilation in [1, 2, 3, 4, 5, 6, 7, 8, 9, 27]:
                            with self.subTest(B=B, T=T, C=C, kernel_size=kernel_size, dilation=dilation):
                                self.__test_conv1d_incremental(kernel_size, dilation, T, B, C)
