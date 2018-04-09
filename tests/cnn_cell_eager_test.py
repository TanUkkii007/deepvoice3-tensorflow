import tensorflow as tf
from tensorflow.python.util import nest
import numpy as np
import math
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.modules import Conv1dGLU
from deepvoice3_tensorflow.cnn_cell import MultiCNNCell


@composite
def btc_tensor(draw, b_size=integers(1, 5), t_size=integers(2, 20), c_size=integers(1, 10), elements=integers(-1, 1)):
    b = draw(b_size)
    t = draw(t_size)
    c = draw(c_size)
    btc = draw(arrays(dtype=np.float32, shape=[b, t, c], elements=elements))
    return (b, t, c, btc)


class MultiCNNCellTest(tf.test.TestCase):

    @given(btc_tensor=btc_tensor(), kernel_size=integers(2, 9), dilation=integers(1, 27))
    @settings(max_examples=10, timeout=unlimited)
    def test_multi_cnn_cell(self, btc_tensor, kernel_size, dilation):
        tf.set_random_seed(1234567)
        B, T, C, btc = btc_tensor
        assume(C % 2 == 0)
        # kernel_initializer = tf.truncated_normal_initializer(
        #     mean=0.,
        #     stddev=math.sqrt(4.0 / (float(kernel_size) * C)),
        #     seed=123)

        def one_tenth_initializer(length):
            half = length // 2
            return tf.constant_initializer(np.stack([0.1 * -1 * np.ones(half), 0.1 * np.ones(half)]).reshape(length, order='F'))
        conv1dGLU_cells = [Conv1dGLU(c, 2 + c, kernel_size,
                                     dropout=1, dilation=dilation,
                                     residual=False,
                                     kernel_initializer=one_tenth_initializer(c * 2*(c+2) * kernel_size),
                                     is_incremental=False) for c in [C, C + 2, C + 4]]

        multiConv1dGLU = MultiCNNCell(conv1dGLU_cells, is_incremental=False)

        out = multiConv1dGLU.apply(tf.constant(btc))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     out = sess.run(out)

        conv1dGLU_incremental_cells = [Conv1dGLU(c, 2 + c, kernel_size,
                                                 dropout=1, dilation=dilation,
                                                 residual=False,
                                                 kernel_initializer=one_tenth_initializer(c * 2*(c+2) * kernel_size),
                                                 is_incremental=True) for c in [C, C + 2, C + 4]]
        multiConv1dGLU_incremental = MultiCNNCell(conv1dGLU_incremental_cells, is_incremental=True)
        initial_states = multiConv1dGLU_incremental.zero_state(B, tf.float32)

        def condition(time, unused_inputs, unused_state, unused_outputs):
            return tf.less(time, T)

        def body(time, inputs, state, outputs):
            btc_one = tf.reshape(inputs[:, time, :], shape=(B, -1, C))
            out_online, next_states = multiConv1dGLU_incremental.apply(btc_one, state)
            return (time + 1, inputs, next_states, outputs.write(time, out_online))

        time = tf.constant(0)
        btc = tf.constant(btc)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=T, element_shape=tf.TensorShape([B, 1, C + 6]))
        _, _, _, out_online_ta = tf.while_loop(condition, body, (time, btc, initial_states, outputs_ta))
        out_online = nest.map_structure(lambda ta: ta.stack(), out_online_ta)

        out_online = tf.squeeze(out_online, axis=2)
        out_online = tf.transpose(out_online, perm=(1, 0, 2))

        # with self.test_session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     out_online = sess.run(out_online)

        self.assertAllClose(out, out_online)


if __name__ == '__main__':
    tf.enable_eager_execution()
    tf.test.main()