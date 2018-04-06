import tensorflow as tf
import numpy as np
import math
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.modules import Conv1dGLU


@composite
def btc_tensor(draw, b_size=integers(1, 5), t_size=integers(1, 20), c_size=integers(1, 10), elements=integers(-5, 5)):
    b = draw(b_size)
    t = draw(t_size)
    c = draw(c_size)
    btc = draw(arrays(dtype=np.float32, shape=[b, t, c], elements=elements))
    return (b, t, c, btc)


class Conv1dGLUTest(tf.test.TestCase):

    @given(btc_tensor=btc_tensor(), kernel_size=integers(2, 9), dilation=integers(1, 27))
    @settings(max_examples=10, timeout=unlimited)
    def test_conv1dGLU(self, btc_tensor, kernel_size, dilation):
        tf.set_random_seed(123)
        B, T, C, btc = btc_tensor
        assume(C % 2 == 0)
        kernel_initializer = tf.truncated_normal_initializer(
            mean=0.,
            stddev=math.sqrt(4.0 / (float(kernel_size) * C)),
            seed=123)
        conv1dGLU = Conv1dGLU(C, 2 * C, kernel_size,
                              dropout=0.0, dilation=dilation,
                              kernel_initializer=kernel_initializer,
                              is_incremental=False)
        btc_pf = tf.placeholder(dtype=tf.float32, shape=[B, T, C])
        out = conv1dGLU.apply(btc_pf)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_conv = sess.run(out, feed_dict={btc_pf: btc})

        btc_one_pf = tf.placeholder(dtype=tf.float32, shape=[B, 1, C])
        buffer_size = kernel_size + (kernel_size - 1) * (dilation - 1)
        input_buffer_pf = tf.placeholder(dtype=tf.float32, shape=[B, buffer_size, C])

        conv1dGLU_incremental = Conv1dGLU(C, 2 * C, kernel_size,
                                          dropout=0.0, dilation=dilation,
                                          kernel_initializer=kernel_initializer,
                                          is_incremental=True)
        out_online, next_input_buffer = conv1dGLU_incremental.apply(btc_one_pf, input_buffer=input_buffer_pf)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_conv_online = []
            input_buffer = np.zeros(shape=[B, buffer_size, C], dtype=np.float32)

            for t in range(T):
                result, input_buffer = sess.run([out_online, next_input_buffer], feed_dict={
                    btc_one_pf: btc[:, t, :].reshape(B, -1, C),
                    input_buffer_pf: input_buffer
                })
                output_conv_online += [result]

        output_conv_online = np.stack(output_conv_online).squeeze(axis=2)
        output_conv_online = output_conv_online.transpose((1, 0, 2))

        self.assertAllClose(output_conv, output_conv_online)
