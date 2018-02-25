import tensorflow as tf


class CNNCell(tf.layers.Layer):
    '''
    state propagation implementation based on RNNCell
    '''

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.

        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    @property
    def is_incremental(self):
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        raise NotImplementedError("Abstract method")

    @property
    def require_state(self):
        return self.is_incremental


class MultiCNNCell(CNNCell):
    def __init__(self, cells, is_incremental):
        super(MultiCNNCell, self).__init__()
        assert all([cell.is_incremental == is_incremental for cell in cells])
        self._cells = cells
        self._is_incremental = is_incremental

    @property
    def state_size(self):
        return [cell.state_size for cell in self._cells]

    @property
    def output_size(self):
        return self._cells[-1].output_size

    @property
    def is_incremental(self):
        return self._is_incremental

    def zero_state(self, batch_size, dtype):
        return [cell.zero_state(batch_size, dtype) for cell in self._cells]

    @property
    def require_state(self):
        return any([c.require_state for c in self._cells])

    def call(self, inputs, state=None):
        new_states = []
        current_input = inputs
        for i, cell in enumerate(self._cells):
            with tf.variable_scope("cell_%d" % i):
                if cell.require_state:
                    current_state = state[i]
                    current_input, new_state = cell(current_input, current_state)
                    new_states.append(new_state)
                else:
                    current_input = cell.apply(current_input)
                    new_states.append(None)
        return current_input if not self.require_state else (current_input, new_states)

