import tensorflow as tf
from .ops import causal_conv, Conv1dIncremental
from .weight_normalization import weight_normalization
from .cnn_cell import CNNCell
from .positional_concoding import PositionalEncoding
import math


def linear(inputs, in_features, out_features, dropout=1, weight_initializer_seed=None):
    module = Linear(in_features, out_features, dropout, weight_initializer_seed)
    return module.apply(inputs)


class Linear(tf.layers.Layer):
    """Weight-normalized Linear layer (input: B x T x C)"""

    def __init__(self, in_features, out_features, dropout=1, weight_initializer_seed=None, trainable=True,
                 name=None, **kwargs):
        super(Linear, self).__init__(name=name, trainable=trainable, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        stddev = math.sqrt(dropout / in_features)
        self.weight_initializer = tf.truncated_normal_initializer(mean=0,
                                                                  stddev=stddev, seed=weight_initializer_seed)

    def build(self, input_shape):
        if not self.built:
            # no dependency on input_shape
            weight = self.add_variable("weight", shape=(self.in_features, self.out_features),
                                       initializer=self.weight_initializer,
                                       trainable=False)
            self.normalized_weight = weight_normalization(weight)
            self.bias = self.add_variable("bias", shape=(self.out_features), initializer=tf.zeros_initializer())
            self.built = True

    def call(self, inputs, training=False):
        if inputs.shape.ndims == 2:
            return tf.matmul(inputs, self.normalized_weight) + self.bias
        else:
            return tf.einsum("btc,ce->bte", inputs, self.normalized_weight)


class SinusoidalEncodingEmbedding(tf.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, trainable=True, name=None, **kwargs):
        super(SinusoidalEncodingEmbedding, self).__init__(name=name, trainable=trainable, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        initial_pe = PositionalEncoding.initial_value(self.num_embeddings, self.embedding_dim, position_rate=1.0)
        self.weight = self.add_variable("weight", shape=initial_pe.shape, dtype=tf.float32,
                                        initializer=tf.constant_initializer(initial_pe.value,
                                                                            verify_shape=True))
        self.built = True

    def call(self, x, w=1.0):
        encoded = PositionalEncoding(self.weight, self.num_embeddings, self.embedding_dim).sinusoidal_encode(w)
        return tf.nn.embedding_lookup(encoded, x)


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


class Conv1d(CNNCell):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation, is_incremental,
                 is_training,
                 dropout=1, kernel_initializer=None, bias_initializer=None,
                 normalize_weight=False, trainable=True, name=None):
        super(Conv1d, self).__init__(name=name, trainable=trainable)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = 0 if is_incremental else (kernel_size - 1) * dilation
        self.activation = activation
        self._is_incremental = is_incremental
        self.is_training = is_training
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.normalize_weight = normalize_weight

    @property
    def state_size(self):
        kernel_size = self.kernel_size
        buffer_size = kernel_size + (kernel_size - 1) * (self.dilation - 1)
        return tf.TensorShape([None, buffer_size, self.in_channels])

    @property
    def output_size(self):
        return self.out_channels

    @property
    def is_incremental(self):
        return self._is_incremental

    def zero_state(self, batch_size, dtype):
        shape = self.state_size
        shape = shape.merge_with(tf.TensorShape([batch_size, None, None])) if not shape.is_fully_defined() else shape
        return tf.zeros(shape=shape, dtype=dtype)

    def build(self, input_shape):
        assert input_shape[2].value == self.in_channels
        kernel_size = self.kernel_size
        in_channels = self.in_channels
        out_channels = self.out_channels

        std_factor = 4.0 if self.normalize_weight else 1.0
        std = math.sqrt((std_factor * self.dropout) / (float(kernel_size) * in_channels))
        kernel_initializer = tf.truncated_normal_initializer(mean=0.,
                                                             stddev=std) if self.kernel_initializer is None else self.kernel_initializer
        kernel_trainability = not self.normalize_weight
        self.kernel = self.add_variable("kernel",
                                        shape=[kernel_size, in_channels, out_channels],
                                        initializer=kernel_initializer,
                                        trainable=kernel_trainability)
        if self.normalize_weight:
            self.kernel = weight_normalization(self.kernel)
        bias_initializer = tf.zeros_initializer() if self.bias_initializer is None else self.bias_initializer
        self.bias = self.add_variable("bias",
                                      shape=(1, 1, out_channels),
                                      initializer=bias_initializer)
        self.built = True

    def call(self, inputs, state=None):
        kernel_size = self.kernel_size
        in_channels = self.in_channels
        out_channels = self.out_channels
        padding = self.padding
        input_buffer = state
        if padding > 0:
            inputs = tf.pad(inputs, [[0, 0], [padding, 0], [0, 0]], 'constant')

        if self.is_incremental:
            conv1d_incremental = Conv1dIncremental(tf.transpose(self.kernel, perm=[2, 1, 0]), in_channels,
                                                   out_channels,
                                                   kernel_size, self.dilation)
            conv1d_output, next_input_buffer = conv1d_incremental.apply(inputs, training=self.is_training,
                                                                        input_buffer=input_buffer)
        else:
            conv1d_output = causal_conv(inputs, self.kernel, self.dilation)
        ha = self.activation(conv1d_output + self.bias) if self.activation is not None else (
                conv1d_output + self.bias)
        if self.is_incremental:
            return ha, next_input_buffer
        else:
            return ha


class Conv1dGLU(CNNCell):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 dropout, dilation=1, residual=False, kernel_initializer_seed=None,
                 is_incremental=False, is_training=False, trainable=True, name=None):
        assert in_channels % 2 == 0
        super(Conv1dGLU, self).__init__(name=name, trainable=trainable)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.residual = residual
        self.dilation = dilation
        self._is_incremental = is_incremental
        std_factor = 4.0
        self.kernel_stddev = math.sqrt((std_factor * dropout) / (float(kernel_size) * in_channels))

        self.kernel_initializer = tf.truncated_normal_initializer(mean=0.,
                                                                  stddev=self.kernel_stddev,
                                                                  seed=kernel_initializer_seed)
        self.bias_initializer = tf.zeros_initializer()

        self.convolution = Conv1d(in_channels, 2 * out_channels, kernel_size, dilation, activation=None,
                                  is_incremental=is_incremental,
                                  is_training=is_training,
                                  dropout=dropout,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  normalize_weight=True)

    def build(self, input_shape):
        in_channels_tensor = input_shape[2]
        with tf.control_dependencies([
            tf.assert_equal(self.in_channels, in_channels_tensor)
        ]):
            super(Conv1dGLU, self).build(input_shape)

    def call(self, inputs, input_buffer=None):
        residual = inputs
        x = tf.nn.dropout(inputs, self.dropout)
        # split at C
        splitdim = -1
        if self.is_incremental:
            x, next_input_buffer = self.convolution(inputs, input_buffer)
        else:
            x = self.convolution(inputs)

        a, b = tf.split(x, num_or_size_splits=2, axis=splitdim)
        # apply GLU
        x = a * tf.nn.sigmoid(b)
        # to preserve variance after residual connection, scale by \sqrt{0.5}
        output = (x + residual) * math.sqrt(0.5) if self.residual else x
        if self.is_incremental:
            return output, next_input_buffer
        else:
            return output

    @property
    def state_size(self):
        return self.convolution.state_size

    @property
    def output_size(self):
        return self.out_channels

    @property
    def is_incremental(self):
        return self._is_incremental

    def zero_state(self, batch_size, dtype):
        return self.convolution.zero_state(batch_size, dtype)


def conv1dGLU(input, in_channels, out_channels, kernel_size,
              dropout, dilation=1, residual=False, is_training=False):
    module = Conv1dGLU(in_channels, out_channels, kernel_size,
                       dropout, dilation, residual, is_incremental=False)
    return module.apply(input, training=is_training)


def conv1dGLU_incremental(input, in_channels, out_channels, kernel_size,
                          dropout, input_buffer, dilation=1, residual=False, is_training=False):
    module = Conv1dGLU(in_channels, out_channels, kernel_size,
                       dropout, dilation, residual, is_incremental=True)
    return module.apply(input, input_buffer=input_buffer, training=is_training)
