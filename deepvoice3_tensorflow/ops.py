import tensorflow as tf


def time_to_batch(value, dilation, name='time_to_batch'):
    '''

    :param value: (B, T, C)
    :param dilation:
    :param name:
    :return: (B * dilation, T / dilation, C)
    '''
    with tf.name_scope(name):
        shape = tf.shape(value)
        shape_t, shape_b, shape_c = shape[0], shape[1], shape[2]
        pad_elements = dilation - 1 - (shape_b + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape_c])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape_t * dilation, -1, shape_c])


def batch_to_time(value, dilation, name='batch_to_time'):
    '''

    :param value: (B, T, C)
    :param dilation:
    :param name:
    :return: (B / dilation, T * dilation, C)
    '''
    with tf.name_scope(name):
        shape = tf.shape(value)
        shape_b, shape_t, shape_c = shape[0], shape[1], shape[2]
        prepared = tf.reshape(value, [dilation, -1, shape_c])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])  # (B * T / dilation, dilation, C)
        return tf.reshape(transposed, [tf.div(shape_b, dilation), -1, shape_c])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    '''

    :param value: (B, T, C)
    :param filter_: (filter_width, in_channels, out_channels)
    :param dilation:
    :param name:
    :return:
    '''
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        restored = tf.nn.convolution(value, filter_, padding='VALID', dilation_rate=[dilation])
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        # [batch, out_width, out_channels]
        result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
        return result


# ToDo: do not use tf.layers.Layer. see tf.nn.convolution.
# ToDo: remove bias option
class Conv1dIncremental(tf.layers.Layer):
    def __init__(self, weight, in_channels, out_channels, kernel_size, dilation=1, bias=None, name="conv1d_incremental",
                 trainable=True, **kwargs):
        '''

        :param weight: (out_channels, in_channels, kernel_size)
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param dilation:
        :param bias:
        :param name:
        :param trainable:
        :param kwargs:
        '''
        super(Conv1dIncremental, self).__init__(name=name, trainable=trainable, **kwargs)
        # (out_channels, in_channels, kernel_size)
        self.weight = weight
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.bias = bias

    def build(self, input_shape):
        # input: bsz x len x dim
        self.batch_size = input_shape[0]
        self.shape_c = input_shape[2]
        with tf.control_dependencies(
                [tf.assert_equal(tf.shape(self.weight), (self.out_channels, self.in_channels, self.kernel_size))]):
            super(Conv1dIncremental, self).build(input_shape)

    def call(self, inputs, input_buffer=None, training=False):
        # input: (B, T, C) where T=1
        if training:
            raise RuntimeError('Conv1dIncremental only supports eval mode')
        if input_buffer is None:
            raise ValueError("input_buffer tensor is required")
        kw = self.kernel_size
        dilation = self.dilation
        if kw > 1:
            input_buffer = tf.slice(input_buffer, begin=[0, 1, 0], size=[-1, -1, -1])
            # append next input
            input_buffer = tf.concat(
                [input_buffer,
                 tf.slice(inputs, begin=[0, tf.shape(inputs)[1] - 1, 0], size=[-1, -1, -1])],
                axis=1)
            next_input_buffer = input_buffer
            if dilation > 1:
                input_buffer = input_buffer[:, 0::dilation, :]

        # (out_channels, in_channels, dilation(kernel_size))
        weight = tf.transpose(self.weight, perm=[0, 2, 1])
        # (out_channels, dilation(kernel_size) * in_channels)
        weight = tf.reshape(weight, shape=[self.out_channels, -1])
        # (batch_size, dilation(kernel_size) * in_channels)
        inputs = tf.reshape(input_buffer, shape=[self.batch_size, -1])
        # (batch_size, out_channels)
        output = tf.matmul(inputs, tf.transpose(weight))
        if self.bias is not None:
            output = output + self.bias
        # (batch_size, 1, out_channels)
        output = tf.reshape(output, shape=[self.batch_size, 1, -1])
        return output, next_input_buffer

    def initial_input_buffer(self):
        kw = self.kernel_size
        input_buffer = tf.zeros(shape=[self.batch_size, kw + (kw - 1) * (self.dilation - 1), self.shape_c])
        return input_buffer

