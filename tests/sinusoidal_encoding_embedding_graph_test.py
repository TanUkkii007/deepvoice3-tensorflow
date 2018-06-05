import tensorflow as tf
from deepvoice3.modules import SinusoidalEncodingEmbedding
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers
import numpy as np


class SinusoidalEncodingEmbeddingTest(tf.test.TestCase):

    @given(n_positions=integers(2, 20), dimension=integers(2, 128))
    @settings(max_examples=10, timeout=unlimited)
    def test_weakly_orthogonal(self, n_positions, dimension):
        assume(dimension % 2 == 0)

        see = SinusoidalEncodingEmbedding(n_positions, dimension)
        positions = tf.range(1, n_positions)
        x = see(positions)
        xx = tf.matmul(x, tf.transpose(x))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            xx = sess.run(xx)
            eye = dimension // 2 * np.eye(n_positions)[1:, 1:]
            # if x is orthogonal, xx == eye, but x is not exactly orthogonal so only check the diagonal
            self.assertAllClose(np.diag(eye), np.diag(xx))

    @given(n_positions=integers(2, 20), dimension=integers(2, 128))
    @settings(max_examples=10, timeout=unlimited)
    def test_speaker_rate(self, n_positions, dimension):
        assume(dimension % 2 == 0)

        see = SinusoidalEncodingEmbedding(n_positions, dimension)
        positions = tf.range(1, n_positions)
        positions_half = tf.range(1, n_positions // 2)
        x = see(positions)
        x2 = see(positions_half, w=2)
        x2x = tf.matmul(x, tf.transpose(x2))
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            x2x = sess.run(x2x)
            print(x2x)

if __name__ == '__main__':
    tf.test.main()