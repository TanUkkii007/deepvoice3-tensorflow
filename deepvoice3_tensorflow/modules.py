import tensorflow as tf
from .ops import causal_conv, noncausal_conv, Conv1dIncremental
from .weight_normalization import weight_normalization
from .cnn_cell import CNNCell
from .positional_concoding import PositionalEncoding
import math


class Linear(tf.layers.Layer):
    """Weight-normalized Linear layer (input: B x T x C)"""

    def __init__(self, in_features, out_features, dropout=0.0, weight_initializer=None, bias_initializer=None,
                 trainable=True,
                 name=None, **kwargs):
        super(Linear, self).__init__(name=name, trainable=trainable, **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        stddev = math.sqrt((1.0 - dropout) / in_features)
        self.weight_initializer = weight_initializer if weight_initializer is not None else tf.truncated_normal_initializer(
            mean=0,
            stddev=stddev)
        self.bias_initializer = bias_initializer if bias_initializer is not None else tf.zeros_initializer()

    def build(self, input_shape):
        if not self.built:
            # no dependency on input_shape
            weight = self.add_variable("weight", shape=(self.in_features, self.out_features),
                                       initializer=self.weight_initializer,
                                       trainable=False)
            self.normalized_weight = weight_normalization(weight)
            self.bias = self.add_variable("bias", shape=(self.out_features), initializer=self.bias_initializer)
            self.built = True

    def call(self, inputs, training=False):
        if inputs.shape.ndims == 2:
            return tf.matmul(inputs, self.normalized_weight) + self.bias
        else:
            return tf.einsum("btc,ce->bte", inputs, self.normalized_weight)


class Embedding(tf.layers.Layer):

    def __init__(self, num_embeddings, embedding_dim, stddev=0.01, weight_initializer=None,
                 trainable=True, name=None):
        super(Embedding, self).__init__(name=name, trainable=trainable)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_initializer = weight_initializer if weight_initializer is not None else tf.truncated_normal_initializer(
            mean=0,
            stddev=stddev)

    def build(self, _):
        weight = self.add_variable("weight", shape=(self.num_embeddings, self.embedding_dim),
                                   dtype=tf.float32,
                                   initializer=self.weight_initializer,
                                   trainable=False)
        self.normalized_weight = weight_normalization(weight)
        self.built = True

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.normalized_weight, inputs)


class SinusoidalEncodingEmbedding(tf.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, trainable=True, name=None, **kwargs):
        super(SinusoidalEncodingEmbedding, self).__init__(name=name, trainable=trainable, **kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.initial_pe = PositionalEncoding.initial_value(self.num_embeddings, self.embedding_dim, position_rate=1.0)

    def build(self, input_shape):
        initializer = lambda shape, dtype, partition_info: self.initial_pe.value
        self.weight = self.add_variable("weight", shape=self.initial_pe.shape, dtype=tf.float32,
                                        initializer=initializer)
        self.built = True

    def call(self, x, w=1.0):
        encoded = PositionalEncoding(self.weight, self.num_embeddings, self.embedding_dim).sinusoidal_encode(w)
        return tf.nn.embedding_lookup(encoded.value, x)


class Conv1d(CNNCell):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation, is_incremental,
                 is_training,
                 dropout=0.0, kernel_initializer=None, bias_initializer=None,
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
        # ToDo: Does not work if batch_size is a Tensor
        shape = shape.merge_with(tf.TensorShape([batch_size, None, None])) if not shape.is_fully_defined() else shape
        return tf.zeros(shape=shape, dtype=dtype)

    def build(self, input_shape):
        assert input_shape[2].value == self.in_channels
        kernel_size = self.kernel_size
        in_channels = self.in_channels
        out_channels = self.out_channels

        std_factor = 4.0 if self.normalize_weight else 1.0
        std = math.sqrt((std_factor * (1.0 - self.dropout)) / (float(kernel_size) * in_channels))
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


class NonCausalConv1d(tf.layers.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, activation,
                 dropout=0.0, kernel_initializer=None, bias_initializer=None,
                 normalize_weight=False, trainable=True, name=None):
        super(NonCausalConv1d, self).__init__(name=name, trainable=trainable)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.activation = activation
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.normalize_weight = normalize_weight

    def build(self, input_shape):
        assert input_shape[2].value == self.in_channels
        kernel_size = self.kernel_size
        in_channels = self.in_channels
        out_channels = self.out_channels

        std_factor = 4.0 if self.normalize_weight else 1.0
        std = math.sqrt((std_factor * (1.0 - self.dropout)) / (float(kernel_size) * in_channels))
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

    def call(self, inputs, **kwargs):
        conv1d_output = noncausal_conv(inputs, self.kernel, self.dilation)
        ha = self.activation(conv1d_output + self.bias) if self.activation is not None else (
                conv1d_output + self.bias)
        return ha


class Conv1dGLU(CNNCell):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 dropout, dilation=1, kernel_initializer=None, bias_initializer=None,
                 is_incremental=False, is_training=False, trainable=True, name=None):
        assert in_channels % 2 == 0
        super(Conv1dGLU, self).__init__(name=name, trainable=trainable)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilation = dilation
        self._is_incremental = is_incremental
        std_factor = 4.0
        self.kernel_stddev = math.sqrt((std_factor * (1.0 - dropout)) / (float(kernel_size) * in_channels))

        self.kernel_initializer = kernel_initializer if kernel_initializer is not None else tf.truncated_normal_initializer(
            mean=0.,
            stddev=self.kernel_stddev)
        self.bias_initializer = bias_initializer if bias_initializer is not None else tf.zeros_initializer()

        self.convolution = Conv1d(in_channels, 2 * out_channels, kernel_size, dilation, activation=None,
                                  is_incremental=is_incremental,
                                  is_training=is_training,
                                  dropout=dropout,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  normalize_weight=True)
        self.training = is_training

    def build(self, input_shape):
        in_channels_tensor = input_shape[2]
        with tf.control_dependencies([
            tf.assert_equal(self.in_channels, in_channels_tensor)
        ]):
            self.built = True

    def call(self, inputs, input_buffer=None):
        residual = inputs
        x = tf.layers.dropout(inputs, rate=self.dropout, training=self.training)
        # split at C
        splitdim = -1
        if self.is_incremental:
            x, next_input_buffer = self.convolution(x, input_buffer)
        else:
            x = self.convolution(x)

        a, b = tf.split(x, num_or_size_splits=2, axis=splitdim)
        # apply GLU
        output = a * tf.nn.sigmoid(b)
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


class NonCausalConv1dGLU(tf.layers.Layer):
    """(Dilated) Conv1d + Gated linear unit + (optionally) speaker embedding
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 dropout, dilation=1, kernel_initializer=None, bias_initializer=None,
                 is_incremental=False, is_training=False, trainable=True, name=None):
        assert in_channels % 2 == 0
        super(NonCausalConv1dGLU, self).__init__(name=name, trainable=trainable)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.dilation = dilation
        self._is_incremental = is_incremental
        std_factor = 4.0
        self.kernel_stddev = math.sqrt((std_factor * (1.0 - dropout)) / (float(kernel_size) * in_channels))

        self.kernel_initializer = kernel_initializer if kernel_initializer is not None else tf.truncated_normal_initializer(
            mean=0.,
            stddev=self.kernel_stddev)
        self.bias_initializer = bias_initializer if bias_initializer is not None else tf.zeros_initializer()

        self.convolution = NonCausalConv1d(in_channels, 2 * out_channels, kernel_size, dilation, activation=None,
                                           dropout=dropout,
                                           kernel_initializer=self.kernel_initializer,
                                           bias_initializer=self.bias_initializer,
                                           normalize_weight=True
                                           )
        self.training = is_training

    def build(self, input_shape):
        in_channels_tensor = input_shape[2]
        with tf.control_dependencies([
            tf.assert_equal(self.in_channels, in_channels_tensor)
        ]):
            self.built = True

    def call(self, inputs, input_buffer=None):
        residual = inputs
        x = tf.layers.dropout(inputs, rate=self.dropout, training=self.training)
        # split at C
        splitdim = -1

        x = self.convolution(x)

        a, b = tf.split(x, num_or_size_splits=2, axis=splitdim)
        # apply GLU
        x = a * tf.nn.sigmoid(b)
        # to preserve variance after residual connection, scale by \sqrt{0.5}
        output = (x + residual) * math.sqrt(0.5)
        return output
