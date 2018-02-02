import tensorflow as tf
from .ops import causal_conv, Conv1dIncremental
import math


def embedding(num_embeddings, embedding_dim, inputs, stddev=0.01, name='embedding'):
    '''

    :param num_embeddings:
    :param embedding_dim:
    :param inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
    :param stddev:
    :param name:
    :return: (N, T_in, embedding_dim)
    '''
    embedding_table = tf.get_variable(name, [num_embeddings, embedding_dim], dtype=tf.float32
                                      , initializer=tf.truncated_normal_initializer(stddev=stddev))
    return tf.nn.embedding_lookup(embedding_table, inputs)


def _conv1d(inputs, in_channels, out_channels, kernel_size, dilation, padding, activation, is_incremental, is_training,
            scope,
            input_buffer=None, dropout=0, std_mul=4.0, kernel_initializer=None, bias_initializer=None):
    with tf.variable_scope(scope):
        if padding > 0:
            inputs = tf.pad(inputs, [[0, 0], [padding, 0], [0, 0]], 'constant')
        in_channels_tensor = tf.shape(inputs)[2]
        with tf.control_dependencies([
            tf.assert_equal(in_channels, in_channels_tensor)
        ]):
            std = math.sqrt((std_mul * (1.0 - dropout)) / (float(kernel_size) * in_channels))
            kernel_initializer = tf.truncated_normal_initializer(mean=0.,
                                                                 stddev=std) if kernel_initializer is None else kernel_initializer
            kernel = tf.get_variable("kernel",
                                     shape=[kernel_size, in_channels, out_channels],
                                     initializer=kernel_initializer)
            bias_initializer = tf.zeros_initializer() if bias_initializer is None else bias_initializer
            bias = tf.get_variable("bias",
                                   shape=(1, 1, out_channels),
                                   initializer=bias_initializer)
            if is_incremental:
                conv1d_incremental = Conv1dIncremental(tf.transpose(kernel, perm=[2, 1, 0]), in_channels, out_channels,
                                                       kernel_size, dilation)
                conv1d_output, next_input_buffer = conv1d_incremental.apply(inputs, training=is_training,
                                                                            input_buffer=input_buffer)
            else:
                conv1d_output = causal_conv(inputs, kernel, dilation)

            ha = activation(conv1d_output + bias)

            # ToDo: use weight normalization
            # return tf.layers.batch_normalization(ha, is_training)
            if is_incremental:
                return ha, next_input_buffer
            else:
                return ha


def conv1d(inputs, in_channels, out_channels, kernel_size, dilation, activation, is_training, scope, dropout=0,
           std_mul=4.0, kernel_initializer=None, bias_initializer=None):
    padding = (kernel_size - 1) * dilation
    return _conv1d(inputs, in_channels, out_channels, kernel_size, dilation, padding=padding, activation=activation, is_incremental=False,
                   is_training=is_training, scope=scope,
                   input_buffer=None, dropout=dropout, std_mul=std_mul, kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)


def conv1d_incremental(inputs, in_channels, out_channels, kernel_size, dilation, activation, scope, input_buffer=None,
                       dropout=0,
                       std_mul=4.0, kernel_initializer=None, bias_initializer=None):
    return _conv1d(inputs, in_channels, out_channels, kernel_size, dilation, padding=0, activation=activation,
                   is_incremental=True, is_training=False, scope=scope,
                   input_buffer=input_buffer, dropout=dropout, std_mul=std_mul, kernel_initializer=kernel_initializer,
                   bias_initializer=bias_initializer)


class Conv1dGLU(tf.layers.Layer):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, out_channels, kernel_size,
                 dropout, dilation=1, causal=False, residual=False,
                 trainable=True, name=None):
        super(Conv1dGLU, self).__init__(name=name, trainable=trainable)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.residual = residual
        self.dilation = dilation
        self.causal = causal

    def build(self, input_shape):
        super(Conv1dGLU, self).build(input_shape)

    def call(self, inputs, training=False):
        residual = inputs
        x = tf.nn.dropout(inputs, self.dropout if training else 1)
        splitdim = 1
        x = conv1d(x, 2 * self.out_channels, self.kernel_size, dropout=self.dropout,
                   dilation=self.dilation)
        a, b = tf.split(x, tf.size(x) // 2, axis=splitdim)
        # apply GLU
        x = a * tf.nn.sigmoid(b)
        # to preserve variance after residual connection, scale by \sqrt{0.5}
        return (x + residual) * math.sqrt(0.5) if self.residual else x
