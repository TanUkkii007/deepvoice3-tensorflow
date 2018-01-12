import tensorflow as tf
from deepvoice3_tensorflow.positional_concoding import PositionalEncoding

class PositionalEncodingTest(tf.test.TestCase):

    def test_shift_linearity(self):
        pe = PositionalEncoding(4, 6)
        pe_shifted = PositionalEncoding(4 + 5, 6)
        x = pe._shift_position_internal(5)[1:,:]
        y = pe_shifted.sinusoidal_encoding[-3:,:]
        self.assertAllClose(x, y)

if __name__ == '__main__':
    tf.test.main()