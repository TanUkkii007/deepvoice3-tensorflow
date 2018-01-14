import tensorflow as tf
import numpy as np
from deepvoice3_tensorflow.ops import causal_conv


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

        self.assertAllEqual(result, ref)

    def test_causal_conv(self):
        for n in [5, 11, 21]:
            for dilation in [1, 2, 3, 4, 5, 6]:
                if n - dilation - 1 > 0:
                    with self.subTest(n=n, dilation=dilation):
                        self.__test_causal_conv(n, dilation)

