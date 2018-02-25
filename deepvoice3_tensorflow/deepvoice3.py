import tensorflow as tf
import math
from collections import namedtuple
from .modules import Linear, Conv1dGLU, SinusoidalEncodingEmbedding
from .cnn_cell import CNNCell, MultiCNNCell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism


class ScaledDotProductAttentionMechanism(AttentionMechanism):
    def __init__(self, memory, embed_dim, window_ahead=3, window_backward=1, dropout=1.0, use_key_projection=True,
                 use_value_projection=True):
        '''
        :param memory: (B, src_len, embed_dim)
        :param embed_dim:
        :param window_ahead:
        :param window_backward:
        :param dropout:
        :param use_key_projection:
        :param use_value_projection:
        '''
        if use_key_projection:
            key_projection = Linear(embed_dim, embed_dim, dropout=dropout, name="key_projection")
            self._keys = key_projection(memory)
        else:
            self._keys = memory
        if use_value_projection:
            value_projection = Linear(embed_dim, embed_dim, dropout=dropout, name="value_projection")
            self._values = value_projection(memory)
        else:
            self._values = memory

        self.window_ahead = window_ahead
        self.window_backward = window_backward
        self.dropout = dropout

        self._embed_dim = embed_dim

    @property
    def values(self):
        return self._values

    @property
    def keys(self):
        return self._keys

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def alignment_size(self):
        return self.keys.shape[1].value or tf.shape(self.keys)[1]

    def initial_alignment(self, batch_size, target_length, dtype):
        max_time = self.alignment_size
        size = tf.stack([batch_size, target_length, max_time], axis=0)
        return tf.zeros(size, dtype=dtype)

    # ToDo: use previous_alignments to calculate mask
    def __call__(self, query, previous_alignments=None, mask=None, last_attended=None):
        '''
        :param query: (B, T//r, embed_dim)
        :param previous_alignments:
        :param mask:
        :param last_attended:
        :return:
        '''

        # Q K^\top
        x = tf.matmul(query, self.keys, transpose_b=True)

        x = self._mask(x, mask, last_attended)

        # softmax over last dim
        # (B, tgt_len, src_len)
        shape = tf.shape(x)
        x = tf.nn.softmax(tf.reshape(x, shape=[shape[0] * shape[1], shape[2]]), axis=1)
        x = tf.reshape(x, shape=shape)
        alignment_scores = x

        x = tf.nn.dropout(x, self.dropout)

        x = tf.matmul(x, self.values)

        # scale attention output
        s = tf.cast(tf.shape(self.values)[1], dtype=tf.float32)
        x = x * (s * tf.sqrt(1.0 / s))
        return x, alignment_scores

    def _mask(self, target, mask, last_attended):
        if mask is None or last_attended is None:
            return target
        mask_shape = (target.shape[0].value, 1, -1)
        mask_value = -float("inf")
        mask = tf.reshape(mask, shape=mask_shape)
        mask = mask * mask_value
        backward = last_attended - self.window_backward
        if backward > 0:
            target[:, :, :backward] = mask
        ahead = last_attended + self.window_ahead
        if ahead < target.shape[1].value:
            target[:, :, ahead:] = mask

        return target


class AttentionLayer(tf.layers.Layer):
    def __init__(self, attention_mechanism, conv_channels, dropout=1.0, weight_initializer_seed=None, trainable=True,
                 name=None, **kwargs):
        super(AttentionLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.attention_mechanism = attention_mechanism
        self.conv_channels = conv_channels
        self.dropout = dropout
        self.weight_initializer_seed = weight_initializer_seed

    def build(self, input_shape):
        conv_channels = self.conv_channels
        embed_dim = self.attention_mechanism.embed_dim
        self.query_projection = Linear(conv_channels, embed_dim, dropout=self.dropout,
                                       weight_initializer_seed=self.weight_initializer_seed, name="query_projection")
        self.out_projection = Linear(embed_dim, conv_channels, dropout=self.dropout,
                                     weight_initializer_seed=self.weight_initializer_seed, name="out_projection")

    def call(self, query, mask=None, last_attended=None):
        residual = query

        # attention
        # (B, T//r, embed_dim)
        x = self.query_projection(query)

        x, alignment_scores = self.attention_mechanism(x)

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, alignment_scores


CNNAttentionWrapperState = namedtuple("CNNAttentionWrapperState",
                                      ["cell_state", "time", "alignments", "alignment_history"])


class CNNAttentionWrapper(CNNCell):
    def __init__(self, attention_mechanism, in_channels, out_channels, kernel_size, dilation, dropout,
                 is_incremental, r, kernel_initializer_seed=None, weight_initializer_seed=None, trainable=True,
                 name=None, **kwargs):
        assert in_channels == out_channels
        super(CNNAttentionWrapper, self).__init__(name=name, trainable=trainable, **kwargs)
        self.convolution = Conv1dGLU(in_channels, out_channels, kernel_size,
                                     dropout=dropout, dilation=dilation, residual=False,
                                     kernel_initializer_seed=kernel_initializer_seed,
                                     is_incremental=is_incremental)

        self.attention = AttentionLayer(attention_mechanism, out_channels, dropout,
                                        weight_initializer_seed=weight_initializer_seed)
        self._is_incremental = is_incremental
        self._output_size = out_channels
        self.r = r

    @property
    def is_incremental(self):
        return self._is_incremental

    @property
    def state_size(self):
        return CNNAttentionWrapperState(
            cell_state=self.convolution.state_size,
            time=tf.TensorShape([]),
            alignments=self.attention.attention_mechanism.alignment_size,
            alignment_history=()
        )

    @property
    def output_size(self):
        return self._output_size

    @property
    def require_state(self):
        return True

    def zero_state(self, batch_size, dtype):
        return CNNAttentionWrapperState(cell_state=self.convolution.zero_state(batch_size, dtype),
                                        time=tf.zeros(shape=[], dtype=tf.int32),
                                        alignments=self.attention.attention_mechanism.initial_alignment(batch_size,
                                                                                                        self.r, dtype),
                                        alignment_history=tf.TensorArray(dtype=dtype, size=0, dynamic_size=True))

    def build(self, input_shape):
        self.built = True

    def call(self, query, state=None):
        assert state is not None

        residual = query
        if self.is_incremental:
            query, next_cell_state = self.convolution(query, state.cell_state)
        else:
            query = self.convolution(query)

        # ToDo: properly set mask and last_attended
        output, attention_scores = self.attention(query, mask=None,
                                                  last_attended=None)
        alignment_history = state.alignment_history.write(state.time, attention_scores)
        output = (output + residual) * math.sqrt(0.5)
        if self.is_incremental:
            return output, CNNAttentionWrapperState(next_cell_state, state.time + 1, attention_scores,
                                                    alignment_history)
        else:
            return output, CNNAttentionWrapperState(state.cell_state, state.time, attention_scores,
                                                    alignment_history)


MultiHopAttentionArgs = namedtuple("MultiHopAttentionArgs", ["out_channels", "kernel_size", "dilation", "dropout", "r", "kernel_initializer_seed",
                                        "weight_initializer_seed"])


class MultiHopAttention(CNNCell):
    def __init__(self, attention_mechanism, in_channels, convolutions, is_incremental, trainable=True,
                 name=None, **kwargs):
        super(MultiHopAttention, self).__init__(name=name, trainable=trainable, **kwargs)
        cells = []
        next_in_channels = in_channels
        for i, (out_channels, kernel_size, dilation, dropout, r, kernel_initializer_seed,
                                        weight_initializer_seed) in enumerate(convolutions):
            aw = CNNAttentionWrapper(attention_mechanism, next_in_channels, out_channels, kernel_size,
                                     dilation, dropout, is_incremental, r, kernel_initializer_seed,
                                        weight_initializer_seed)
            next_in_channels = aw.output_size
            cells.append(aw)
        self.layer = MultiCNNCell(cells, is_incremental)
        self._is_incremental = is_incremental

    @property
    def is_incremental(self):
        return self._is_incremental

    @property
    def state_size(self):
        return self.layer.state_size

    @property
    def output_size(self):
        return self.layer.output_size

    def zero_state(self, batch_size, dtype):
        return self.layer.zero_state(batch_size, dtype)

    @property
    def require_state(self):
        return True

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, state):
        return self.layer(inputs, state)

    @staticmethod
    def average_alignment(states):
        return sum([state.alignments for state in states]) / len(states)


