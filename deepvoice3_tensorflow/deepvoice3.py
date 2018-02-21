import tensorflow as tf
import math
from collections import namedtuple
from .modules import Linear, Conv1dGLU
from .cnn_cell import CNNCell, MultiCNNCell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism


class ScaledDotProductAttentionMechanism(AttentionMechanism):
    def __init__(self, memory, embed_dim, window_ahead=3, window_backward=1, dropout=1.0, use_key_projection=True, use_value_projection=True):
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

    # ToDo: use previous_alignments to calculate mask
    def __call__(self, query, previous_alignments=None, mask=None, last_attended=None):
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
    def __init__(self, attention_mechanism, conv_channels, dropout=1.0, trainable=True,
                 name=None, **kwargs):
        super(AttentionLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.attention_mechanism = attention_mechanism
        self.conv_channels = conv_channels
        self.dropout = dropout

    def build(self, input_shape):
        conv_channels = self.conv_channels
        embed_dim = self.attention_mechanism.embed_dim
        self.query_projection = Linear(conv_channels, embed_dim, dropout=self.dropout, name="query_projection")
        self.out_projection = Linear(embed_dim, conv_channels, dropout=self.dropout, name="out_projection")

    def call(self, query, mask=None, last_attended=None):
        residual = query

        # attention
        x = self.query_projection(query)

        x, alignment_scores = self.attention_mechanism(x)

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, alignment_scores


CNNAttentionWrapperState = namedtuple("CNNAttentionWrapperState",
                                      ["cell_state", "time", "alignments", "alignment_history", "mask",
                                       "last_attended"])


class CNNAttentionWrapper(CNNCell):
    def __init__(self, embed_dim, use_key_projection, use_value_projection,
                 window_ahead, window_backward, in_channels, out_channels, kernel_size, dilation, dropout,
                 is_incremental, trainable=True,
                 name=None, **kwargs):
        super(CNNAttentionWrapper, self).__init__(name=name, trainable=trainable, **kwargs)
        self.convolution = Conv1dGLU(in_channels, out_channels, kernel_size,
                                     dropout=dropout, dilation=dilation, residual=False,
                                     is_incremental=is_incremental)

        self.attention = AttentionLayer(out_channels, embed_dim, dropout, use_key_projection, use_value_projection,
                                        window_ahead, window_backward)
        self._is_incremental = is_incremental
        self._output_size = out_channels

    @property
    def is_incremental(self):
        return self._is_incremental

    @property
    def state_size(self):
        alignment_size = None
        return (
            self.convolution.state_size, tf.TensorShape([]), tf.TensorShape([alignment_size, self.attention.embed_dim]),
            None, None, None)

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        alignment_shape = tf.TensorShape([batch_size, None, None]).merge_with(self.state_size[2])
        return CNNAttentionWrapperState(cell_state=self.convolution.zero_state(batch_size, dtype),
                                        time=tf.zeros(shape=[], dtype=tf.int32),
                                        alignments=tf.zeros(shape=alignment_shape, dtype=dtype),
                                        alignment_history=tf.TensorArray(dtype=dtype, size=0, dynamic_size=True),
                                        mask=None,
                                        last_attended=None)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, state):
        query, keys, values = inputs
        residual = query
        if self.is_incremental:
            query, next_cell_state = self.convolution(query, state.cell_state)
        else:
            query = self.convolution(query)

        output, attention_scores = self.attention(AttentionInput(query, keys, values), mask=state.mask,
                                                  last_attended=state.last_attended)
        alignment_history = state.alignment_history.write(state.time, attention_scores)
        output = (output + residual) * math.sqrt(0.5)
        # ToDo: properly set mask and last_attended
        return output, CNNAttentionWrapperState(next_cell_state, state.time + 1, attention_scores, alignment_history,
                                                mask=None, last_attended=None)

