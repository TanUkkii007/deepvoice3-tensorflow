import numpy as np
import tensorflow as tf

class PositionalEncoding(object):
    def __init__(self, value, n_positions, dimension):
        assert dimension % 2 == 0
        assert value.shape.ndims == 2
        assert value.shape[0].value == n_positions
        assert value.shape[1].value == dimension
        self._dimension = dimension
        self._n_positions = n_positions
        self._value = value

    @property
    def dimension(self):
        return self._dimension

    @property
    def n_positions(self):
        return self._n_positions

    def sinusoidal_encode(self, position_rate=1.0):
        odd = self._value[:, 0::2]
        even = self._value[:, 1::2]
        return SinusoidalEncoding.encode(odd, even, self, position_rate)

    @staticmethod
    def initial_value(n_position, dimension, position_rate=1.0):
        def value_at(position, index):
            return position_rate * position / np.power(10000, 2 * (index // 2) / dimension)

        values = [[
            value_at(pos, i) for i in
            range(dimension)
        ] for pos in range(n_position)]
        return PositionalEncoding(tf.constant(np.array(values)), n_position, dimension)


class SinusoidalEncoding(object):
    def __init__(self, odd, even, positional_encoding):
        self.odd = odd
        self.even = even
        self._pe = positional_encoding

    @staticmethod
    def encode(odd, even, positional_encoding, position_rate=1.0):
        odd = position_rate * odd
        even = position_rate * even
        odd = tf.concat([tf.expand_dims(odd[0, :], axis=0), tf.sin(odd[1:, :])], axis=0)
        even = tf.concat([tf.expand_dims(even[0, :], axis=0), tf.cos(even[1:, :])], axis=0)
        return SinusoidalEncoding(odd, even, positional_encoding)

    @property
    def value(self):
        dimension = self._pe.dimension
        odd_idx = [[d] for d in range(0, dimension, 2)]
        even_idx = [[d] for d in range(1, dimension, 2)]
        shape = [self._pe.dimension, self._pe.n_positions] # transposed
        updates = tf.transpose(tf.concat([self.odd, self.even], axis=-1), perm=(1,0))
        return tf.transpose(tf.scatter_nd(indices=odd_idx + even_idx, updates=updates, shape=shape), perm=(1,0))

    def shift_factor(self, shift):
        ''' return shift factor matrix for sinusoidal encoding table
        :math:`\begin{pmatrix}
        \cos(k/a) & \sin(k/a)\\
        -\sin(k/a) & \cos(k/a)
        \end{pmatrix}`

        .. math::
        \begin{pmatrix}
        \mathrm{PE}_{\mathrm{pos} + k, 2i}\\
        \mathrm{PE}_{\mathrm{pos} + k, 2i+1}
        \end{pmatrix} = \begin{pmatrix}
        \cos(k/a) & \sin(k/a)\\
        -\sin(k/a) & \cos(k/a)
        \end{pmatrix}
        \begin{pmatrix}
        \mathrm{PE}_{\mathrm{pos}, 2i}\\
        \mathrm{PE}_{\mathrm{pos}, 2i+1}
        \end{pmatrix}

        :param shift:
        :return: ndarray with `dimension`-length
        '''

        def dimension_constant(i):
            return np.power(10000, 2 * (i // 2) / self._pe.dimension)

        shifts = list(
            [shift / dimension_constant(i) for i in range(self._pe.dimension)])
        shifts[0::2] = [[np.cos(s), np.sin(s)] for s in shifts[0::2]]
        shifts[1::2] = [[-np.sin(s), np.cos(s)] for s in shifts[1::2]]
        return np.array(shifts)

    def shift_n(self, shift):
        odd_even = tf.transpose(tf.stack([self.odd, self.even]), perm=(2, 0, 1))  # (dimension//2, 2, n_position)
        factor = self.shift_factor(shift)
        odd_factor = factor[0::2]
        even_factor = factor[1::2]
        odd_even_factor = np.stack([odd_factor, even_factor]).transpose((1, 0, 2))  # (dimension//2, 2, 2)
        shifted = tf.matmul(tf.constant(odd_even_factor), odd_even)  # (dimension//2, 2, n_position)
        new_odd = tf.transpose(shifted[:, 0, :], perm=(1,0))
        new_even = tf.transpose(shifted[:, 1, :], perm=(1,0))
        return SinusoidalEncoding(new_odd, new_even, self._pe)
