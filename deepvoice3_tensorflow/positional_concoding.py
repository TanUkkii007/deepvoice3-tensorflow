import numpy as np


class PositionalEncoding(object):

    def __init__(self, n_position, dimension, position_rate=1.0):
        assert dimension % 2 == 0
        self.n_position = n_position
        self.dimension = dimension
        self.position_rate = position_rate
        self.argument_table = self.initial_arguments()
        self.sinusoidal_encoding = self.encode_sinusoid(self.argument_table, copy=True)

    def position_shift_factor(self, shift):
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
            return np.power(10000, 2 * (i // 2) / self.dimension)

        shifts = list(
            [shift / dimension_constant(i) for i in range(self.dimension)])
        shifts[0::2] = [[np.cos(s), np.sin(s)] for s in shifts[0::2]]
        shifts[1::2] = [[-np.sin(s), np.cos(s)] for s in shifts[1::2]]
        return np.array(shifts)

    def _shift_position_internal(self, shift):
        odd = self.sinusoidal_encoding[:, 0::2]  # (n_position, dimension//2)
        even = self.sinusoidal_encoding[:, 1::2]  # (n_position, dimension//2)
        odd_even = np.stack([odd, even]).transpose((2, 0, 1))  # (dimension//2, 2, n_position)
        factor = self.position_shift_factor(shift)
        odd_factor = factor[0::2]
        even_factor = factor[1::2]
        odd_even_factor = np.stack([odd_factor, even_factor]).transpose((1, 0, 2))  # (dimension//2, 2, 2)
        shifted = odd_even_factor @ odd_even  # (dimension//2, 2, n_position)
        shifted = shifted.reshape(self.dimension,
                                  self.n_position)  # (dimension, n_position)
        return shifted.transpose()

    def initial_arguments(self):
        ''' Create an argument table of sinusoid function for positional encoding.
        :math:`{PE}_{\mathrm{pos}, 2i} = \sin({pos}/10000^{2i/d_{model}})`

        :param n_position:
        :param dimension:
        :param position_rate:
        :return: ndarray of (n_position, dimension)
        '''

        def argument(position, index):
            return self.position_rate * position / np.power(10000, 2 * (index // 2) / self.dimension)

        arguments = [[
            argument(pos, i) for i in
            range(self.dimension)
        ] for pos in range(self.n_position)]
        return np.array(arguments)

    @staticmethod
    def encode_sinusoid(argument_table, copy=True):
        ''' map arguments with sinusoidal function
        :math:`{PE}_{\mathrm{pos}, 2i} = \sin({pos}/10000^{2i/d_{model}})`
        :math:`{PE}_{\mathrm{pos}, 2i+1} = \cos({pos}/10000^{2i+1/d_{model}})`

        :param argument_table: numpy array with dimension `(n_position, dimension)`
        :param copy:
        :return:
        '''
        argument_table = argument_table.copy() if copy else argument_table
        argument_table[1:, 0::2] = np.sin(argument_table[1:, 0::2])
        argument_table[1:, 1::2] = np.cos(argument_table[1:, 1::2])
        return argument_table
