import tensorflow as tf
from deepvoice3_tensorflow.positional_concoding import PositionalEncoding
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers
import numpy as np


class PositionalEncodingTest(tf.test.TestCase):

    @given(n_positions=integers(2, 256), dimension=integers(2, 128), shift=integers(1, 256))
    @settings(max_examples=10, timeout=unlimited)
    def test_shift_linearity(self, n_positions, dimension, shift):
        '''
        test position shift linearity of positional encoding.
        This property holds except for the 0-th position.
        :return:
        '''
        assume(dimension % 2 == 0)

        pe = PositionalEncoding.initial_value(n_positions, dimension)
        pe_shifted = PositionalEncoding.initial_value(n_positions + shift, dimension)
        x = pe.sinusoidal_encode().shift_n(shift).value[1:,:] # drop position 0
        y = pe_shifted.sinusoidal_encode().value[shift:,:]
        y = y[1:, :] # drop position 0
        with self.test_session() as sess:
            x, y = sess.run([x, y])
            print("max diff", np.max(np.abs(y - x)))
            self.assertAllClose(x, y, atol=1e-4)

if __name__ == '__main__':
    tf.test.main()