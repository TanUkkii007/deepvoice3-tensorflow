import tensorflow as tf
import numpy as np
import uuid
from hypothesis import given, settings, unlimited
from hypothesis.strategies import integers
from deepvoice3_tensorflow.modules import conv1d, conv1d_incremental


def curried_leaky_relu(alpha):
    return lambda tensor: tf.nn.leaky_relu(tensor, alpha)

class ModuleTest(tf.test.TestCase):

    @given(B=integers(1, 3), T=integers(10, 30), C=integers(1, 4), kernel_size=integers(2, 9), dilation=integers(1, 27))
    @settings(max_examples=10, timeout=unlimited)
    def test_conv1d(self, kernel_size, dilation, T, B, C):
        with tf.variable_scope(str(uuid.uuid4())):
            bct_value = np.zeros(shape=[B, C, T], dtype=np.float32) + (
                    np.zeros(shape=[C, T], dtype=np.float32) + np.arange(0, T, dtype=np.float32))
            btc_value = bct_value.transpose((0, 2, 1))
            # B, T, C
            btc = tf.placeholder(dtype=tf.float32, shape=[B, T, C])

            kernel_initializer = tf.constant_initializer(np.arange(0, kernel_size * C * 2*C))
            out = conv1d(btc, C, 2 * C, kernel_size, dilation, curried_leaky_relu(0.5), is_training=False,
                         scope="conv1d", kernel_initializer=kernel_initializer)
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                output_causal_conv = sess.run(out, feed_dict={btc: btc_value})


            btc_one = tf.placeholder(dtype=tf.float32, shape=[B, 1, C])
            output_conv_online = []
            buffer_size = kernel_size + (kernel_size - 1) * (dilation - 1)
            input_buffer_pf = tf.placeholder(dtype=tf.float32, shape=[B, buffer_size, C])
            input_buffer = np.zeros(shape=[B, buffer_size, C], dtype=np.float32)
            output_conv, next_input_buffer = conv1d_incremental(btc_one, C, 2 * C, kernel_size=kernel_size,
                                                                dilation=dilation,
                                                                activation=curried_leaky_relu(0.5),
                                                                scope="conv1d_incremental",
                                                                input_buffer=input_buffer_pf,
                                                                kernel_initializer=kernel_initializer)
            with self.test_session() as sess:
                sess.run(tf.global_variables_initializer())
                for t in range(T):
                    result, input_buffer = sess.run([output_conv, next_input_buffer],
                                                    feed_dict={
                                                        btc_one: btc_value[:, t, :].reshape(B, -1, C),
                                                        input_buffer_pf: input_buffer
                                                    })
                    output_conv_online += [result]

            output_conv_online = np.stack(output_conv_online).squeeze(axis=2)
            output_conv_online = output_conv_online.transpose((1, 0, 2))

            self.assertAllEqual(output_causal_conv, output_conv_online)
