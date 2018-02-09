import tensorflow as tf
import math
from .modules import Linear


class AttentionLayer(tf.layers.Layer):
    def __init__(self, conv_channels, embed_dim, dropout=1, use_key_projection=True, use_value_projection=True,
                 window_ahead=3, window_backward=1, trainable=True,
                 name=None):
        super(AttentionLayer, self).__init__(name=name, trainable=trainable)
        self.conv_channels = conv_channels
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.use_key_projection = use_key_projection
        self.use_value_projection = use_value_projection

        self.key_projection = None
        self.value_projection = None
        self.window_ahead = window_ahead
        self.window_backward = window_backward

    def build(self, input_shape):
        conv_channels = self.conv_channels
        embed_dim = self.embed_dim
        with tf.variable_scope("query_projection"):
            self.query_projection = Linear(conv_channels, embed_dim, dropout=self.dropout)
        if self.use_key_projection:
            with tf.variable_scope("key_projection"):
                self.key_projection = Linear(embed_dim, embed_dim, dropout=self.dropout)
        if self.use_value_projection:
            with tf.variable_scope("value_projection"):
                self.value_projection = Linear(embed_dim, embed_dim, dropout=self.dropout)

        with tf.variable_scope("out_projection"):
            self.out_projection = Linear(embed_dim, conv_channels, dropout=self.dropout)


    def call(self, query, encoder_out=None, mask=None, last_attended=None, training=False):
        keys, values = encoder_out
        residual = query
        if self.use_value_projection:
            values = self.value_projection.apply(values)
        if self.use_key_projection:
            keys = self.key_projection.apply(keys)

        # attention
        x = self.query_projection(query)
        # Q K^\top
        x = tf.matmul(x, keys, transpose_b=True)

        x = self._mask(x, mask, last_attended)

        # softmax over last dim
        # (B, tgt_len, src_len)
        shape = tf.shape(x)
        x = tf.nn.softmax(tf.reshape(x, shape=[shape[0] * shape[1], shape[2]]), axis=1)
        x = tf.reshape(x, shape=shape)
        attention_scores = x

        x = tf.nn.dropout(x, self.dropout)

        x = tf.matmul(x, values)

        # scale attention output
        s = tf.cast(tf.shape(values)[1], dtype=tf.float32)
        x = x * (s * tf.sqrt(1.0 / s))

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, attention_scores

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

