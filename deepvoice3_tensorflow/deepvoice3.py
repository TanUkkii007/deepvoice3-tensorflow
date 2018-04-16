import tensorflow as tf
import math
from collections import namedtuple
from .modules import Linear, Embedding, Conv1d, NonCausalConv1d, Conv1dGLU, NonCausalConv1dGLU, \
    SinusoidalEncodingEmbedding
from .cnn_cell import CNNCell, MultiCNNCell
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import AttentionMechanism
from tensorflow.python.util import nest


class Encoder(tf.layers.Layer):
    def __init__(self, n_vocab, embed_dim, embedding_weight_std=0.1,
                 convolutions=((64, 5, .1),) * 7,  # ToDo: use named tuple
                 dropout=0.1,
                 training=False,
                 trainable=True,
                 name=None, **kwargs):
        super(Encoder, self).__init__(name=name, trainable=trainable, **kwargs)
        self.dropout = dropout
        self.training = training
        self.embed_tokens = Embedding(n_vocab, embed_dim, embedding_weight_std)
        in_channels = embed_dim
        adjustion_layer = NonCausalConv1d(in_channels, convolutions[0][0], kernel_size=1, dilation=1,
                                          activation=tf.nn.relu)
        last_conv = NonCausalConv1d(convolutions[-1][0], embed_dim, kernel_size=1, dilation=1, activation=None)
        self.convolutions = [adjustion_layer] + [
            NonCausalConv1dGLU(out_channels, out_channels, kernel_size, dropout, dilation) for
            (out_channels, kernel_size, dilation) in convolutions] + [last_conv]

    def build(self, _):
        self.built = True

    def call(self, text_sequences, text_positions=None):
        x = self.embed_tokens(text_sequences)
        x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        input_embedding = x
        keys = x
        # use normal convolution instead of causal convolution
        for conv in self.convolutions:
            keys = conv(keys)

        # add output to input embedding for attention
        values = (keys + input_embedding) + math.sqrt(0.5)
        return keys, values

    def register_metrics(self):
        self.embed_tokens.register_metrics()
        for c in self.convolutions:
            c.register_metrics()


class ScaledDotProductAttentionMechanism(AttentionMechanism):
    def __init__(self, keys, values, embed_dim, window_ahead=3, window_backward=1, dropout=0.1, use_key_projection=True,
                 use_value_projection=True, key_projection_weight_initializer=None,
                 key_projection_bias_initializer=None, value_projection_weight_initializer=None,
                 value_projection_bias_initializer=None, training=False):
        '''
        :param memory: (B, src_len, embed_dim)
        :param embed_dim:
        :param window_ahead:
        :param window_backward:
        :param dropout:
        :param use_key_projection:
        :param use_value_projection:
        '''
        self.use_key_projection = use_key_projection
        if use_key_projection:
            self.key_projection = Linear(embed_dim, embed_dim, dropout=dropout,
                                         weight_initializer=key_projection_weight_initializer,
                                         bias_initializer=key_projection_bias_initializer, name="key_projection")
            self._keys = self.key_projection(keys)
        else:
            self._keys = keys
        self.use_value_projection = use_value_projection
        if use_value_projection:
            self.value_projection = Linear(embed_dim, embed_dim, dropout=dropout,
                                           weight_initializer=value_projection_weight_initializer,
                                           bias_initializer=value_projection_bias_initializer, name="value_projection")
            self._values = self.value_projection(values)
        else:
            self._values = values

        self.window_ahead = window_ahead
        self.window_backward = window_backward
        self.dropout = dropout

        self._embed_dim = embed_dim
        self.training = training

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

    def __call__(self, query, memory_mask=None, query_mask=None):
        '''
        :param query: (B, T//r, embed_dim)
        :param mask: (B, T_memory)
        :return:
        '''

        # Q K^\top
        x = tf.matmul(query, self.keys, transpose_b=True)

        scaling_factor = 1.0 / math.sqrt(self._embed_dim)
        x = x * scaling_factor

        x = self._memory_mask(x, memory_mask)
        x = self._query_mask(x, query_mask)

        # softmax over last dim
        # (B, T_query, T_memory)
        x = tf.nn.softmax(x, axis=-1)
        alignment_scores = x

        x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        x = tf.matmul(x, self.values)

        # scale attention output
        s = tf.cast(tf.shape(self.values)[1], dtype=tf.float32)
        x = x / tf.sqrt(s)
        return x, alignment_scores

    def register_metrics(self):
        if self.use_key_projection:
            self.key_projection.register_metrics()
        if self.use_value_projection:
            self.value_projection.register_metrics()

    def _memory_mask(self, qk, memory_mask):
        if memory_mask is None:
            return qk
        # reshape mask to (B, 1, T_memory) to broadcast
        mask = tf.expand_dims(memory_mask, axis=1)
        return qk + mask

    def _query_mask(self, qk, query_mask):
        if query_mask is None:
            return qk
        # reshape mask to (B, T_query, 1) to broadcast
        mask = tf.expand_dims(query_mask, axis=2)
        return qk + mask


class AttentionLayer(tf.layers.Layer):
    def __init__(self, attention_mechanism, conv_channels, dropout=1.0, query_projection_weight_initializer=None,
                 out_projection_weight_initializer=None, trainable=True,
                 name=None, **kwargs):
        super(AttentionLayer, self).__init__(name=name, trainable=trainable, **kwargs)
        self.attention_mechanism = attention_mechanism
        self.conv_channels = conv_channels
        self.dropout = dropout
        self.query_projection_weight_initializer = query_projection_weight_initializer
        self.out_projection_weight_initializer = out_projection_weight_initializer

    def build(self, input_shape):
        conv_channels = self.conv_channels
        embed_dim = self.attention_mechanism.embed_dim
        self.query_projection = Linear(conv_channels, embed_dim, dropout=self.dropout,
                                       weight_initializer=self.query_projection_weight_initializer,
                                       name="query_projection")
        self.out_projection = Linear(embed_dim, conv_channels, dropout=self.dropout,
                                     weight_initializer=self.out_projection_weight_initializer, name="out_projection")

    def call(self, query, memory_mask=None, query_mask=None):
        residual = query

        # attention
        # (B, T//r, embed_dim)
        x = self.query_projection(query)

        x, alignment_scores = self.attention_mechanism(x, memory_mask=memory_mask, query_mask=query_mask)

        # project back
        x = self.out_projection(x)
        x = (x + residual) * math.sqrt(0.5)
        return x, alignment_scores

    def register_metrics(self):
        self.query_projection.register_metrics()
        self.out_projection.register_metrics()
        self.attention_mechanism.register_metrics()


CNNAttentionWrapperState = namedtuple("CNNAttentionWrapperState",
                                      ["cell_state", "time", "alignments", "alignment_history"])

CNNAttentionWrapperInput = namedtuple("CNNAttentionWrapperInput", ["query", "frame_pos_embed"])


class CNNAttentionWrapper(CNNCell):
    def __init__(self, attention_mechanism, in_channels, out_channels, kernel_size, dilation, dropout,
                 is_incremental, r, memory_mask=None, query_mask=None, kernel_initializer=None, query_projection_weight_initializer=None,
                 out_projection_weight_initializer=None,
                 training=False,
                 trainable=True,
                 name=None, **kwargs):
        super(CNNAttentionWrapper, self).__init__(name=name, trainable=trainable, **kwargs)
        # To support residual connection in_channels == out_channels is necessary.
        assert in_channels == out_channels
        self.convolution = Conv1dGLU(in_channels, out_channels, kernel_size,
                                     dropout=dropout, dilation=dilation,
                                     kernel_initializer=kernel_initializer,
                                     is_incremental=is_incremental,
                                     is_training=training)

        self.attention = AttentionLayer(attention_mechanism, out_channels, dropout,
                                        query_projection_weight_initializer=query_projection_weight_initializer,
                                        out_projection_weight_initializer=out_projection_weight_initializer)
        self._is_incremental = is_incremental
        self._output_size = out_channels
        self.r = r
        self.memory_mask = memory_mask
        self.query_mask = query_mask
        self._collect_metrics = training

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
        _, frame_pos_embed_shape = input_shape
        assert frame_pos_embed_shape[-1].value == self.convolution.output_size
        self.built = True

    def call(self, cnn_attention_wrapper_input, state=None):
        assert state is not None

        if isinstance(cnn_attention_wrapper_input, CNNAttentionWrapperInput):
            query, frame_pos_embed = cnn_attention_wrapper_input
        else:
            query, frame_pos_embed = cnn_attention_wrapper_input, None

        residual = query
        if self.is_incremental:
            query, next_cell_state = self.convolution(query, state.cell_state)
        else:
            query = self.convolution(query)

        query = query if frame_pos_embed is None else query + frame_pos_embed
        if self._collect_metrics:
            tf.summary.histogram("query", query)

        output, attention_scores = self.attention(query, memory_mask=self.memory_mask, query_mask=self.query_mask)
        alignment_history = state.alignment_history.write(state.time, attention_scores)
        output = (output + residual) * math.sqrt(0.5)
        if self.is_incremental:
            return CNNAttentionWrapperInput(output, frame_pos_embed), CNNAttentionWrapperState(next_cell_state,
                                                                                               state.time + 1,
                                                                                               attention_scores,
                                                                                               alignment_history)
        else:
            return CNNAttentionWrapperInput(output, frame_pos_embed), CNNAttentionWrapperState(state.cell_state,
                                                                                               state.time,
                                                                                               attention_scores,
                                                                                               alignment_history)

    def register_metrics(self):
        self.convolution.register_metrics()
        self.attention.register_metrics()


MultiHopAttentionArgs = namedtuple("MultiHopAttentionArgs", ["out_channels", "kernel_size", "dilation", "dropout"])


class MultiHopAttention(CNNCell):
    def __init__(self, attention_mechanism, in_channels, convolutions, r, is_incremental,
                 memory_mask=None, query_mask=None, kernel_initializer=None, query_projection_weight_initializer=None,
                 out_projection_weight_initializer=None,
                 training=False,
                 trainable=True,
                 name=None, **kwargs):
        super(MultiHopAttention, self).__init__(name=name, trainable=trainable, **kwargs)
        cells = []
        next_in_channels = in_channels
        for i, (out_channels, kernel_size, dilation, dropout) in enumerate(convolutions):
            aw = CNNAttentionWrapper(attention_mechanism, next_in_channels, out_channels, kernel_size,
                                     dilation, dropout, is_incremental, r, memory_mask, query_mask, kernel_initializer,
                                     query_projection_weight_initializer,
                                     out_projection_weight_initializer,
                                     training)
            next_in_channels = aw.output_size
            cells.append(aw)
        self._layers = cells
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

    def register_metrics(self):
        for c in self._layers:
            c.register_metrics()


class DecoderPreNetCNNArgs(namedtuple("DecoderPreNetCNNArgs", ["out_channels", "kernel_size", "dilation"])):
    pass


class DecoderPreNetCNN(CNNCell):
    def __init__(self, in_channels, params, dropout=0.1, is_incremental=False,
                 kernel_initializer=None, bias_initializer=None, training=False,
                 trainable=True, name=None, **kwargs):
        super(DecoderPreNetCNN, self).__init__(name=name, trainable=trainable, **kwargs)
        adjust_layer = Conv1d(in_channels, params[0].out_channels, kernel_size=1, dilation=1, activation=tf.nn.relu,
                              is_incremental=is_incremental, is_training=training,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer, normalize_weight=True)
        layers = [adjust_layer]
        in_channels = adjust_layer.out_channels
        for out_channels, kernel_size, dilation in params:
            conv = Conv1dGLU(in_channels, out_channels, kernel_size,
                             dropout=dropout, dilation=dilation,
                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                             is_incremental=is_incremental, is_training=training)
            layers.append(conv)
            in_channels = conv.out_channels

        self._layers = layers
        self.layer = MultiCNNCell(layers, is_incremental)

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

    def build(self, _):
        self.built = True

    def call(self, inputs, **kwargs):
        return self.layer(inputs, **kwargs)

    def register_metrics(self):
        for c in self._layers:
            c.register_metrics()


class PositionWiseFFN(CNNCell):
    '''
    Position-wise feed forward network based on CNN with kernel size = 1
    '''

    def __init__(self, in_channels, out_channels, dropout=0.1, is_incremental=False,
                 kernel_initializer=None, bias_initializer=None, training=False,
                 normalize_weight=True,
                 trainable=True, name=None, **kwargs):
        super(PositionWiseFFN, self).__init__(name=name, trainable=trainable, **kwargs)
        self.layer = Conv1d(in_channels, out_channels, kernel_size=1, dilation=1, activation=tf.nn.relu,
                            is_incremental=is_incremental, is_training=training,
                            dropout=dropout,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer, normalize_weight=normalize_weight)

        self._is_incremental = is_incremental
        self.dropout = dropout
        self.training = training

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

    def build(self, _):
        self.built = True

    def call(self, inputs, state=None, **kwargs):
        x = tf.layers.dropout(inputs, rate=self.dropout, training=self.training)
        return self.layer(x, state=state, **kwargs)

    def register_metrics(self):
        self.layer.register_metrics()


class DecoderPreNetArgs(namedtuple("DecoderPreNetArgs", ["out_channels"])):
    pass


class DecoderPreNet(CNNCell):

    def __init__(self, in_channels, params, dropout=0.1, is_incremental=False,
                 kernel_initializer=None, bias_initializer=None, training=False,
                 trainable=True, name=None, **kwargs):
        super(DecoderPreNet, self).__init__(name=name, trainable=trainable, **kwargs)
        adjust_layer = PositionWiseFFN(in_channels, params[0].out_channels,
                                       dropout=dropout,
                                       is_incremental=is_incremental, training=training,
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer, normalize_weight=True)
        layers = [adjust_layer]
        in_channels = adjust_layer.output_size
        for p in params:
            conv = PositionWiseFFN(in_channels, p.out_channels,
                                   dropout=dropout,
                                   kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                   is_incremental=is_incremental, training=training, normalize_weight=True)
            layers.append(conv)
            in_channels = conv.output_size

        self._layers = layers
        self.layer = MultiCNNCell(layers, is_incremental)

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

    def build(self, _):
        self.built = True

    def call(self, inputs, state=None, **kwargs):
        return self.layer(inputs, state=state, **kwargs)

    def register_metrics(self):
        for c in self._layers:
            c.register_metrics()


class Decoder(tf.layers.Layer):
    def __init__(self, embed_dim, in_dim=80, r=5, max_positions=512,
                 preattention=(DecoderPreNetArgs(128),) * 4,
                 mh_attentions=(MultiHopAttentionArgs(128, 5, 1, 0.1),) * 4, dropout=0.1,
                 use_memory_mask=False,
                 use_query_mask=False,
                 query_position_rate=1.0,
                 key_position_rate=1.29,
                 max_decoder_steps=200,
                 min_decoder_steps=10,
                 is_incremental=False,
                 prenet_weight_initializer=None,
                 prenet_bias_initializer=None,
                 attention_key_projection_weight_initializer=None,
                 attention_key_projection_bias_initializer=None,
                 attention_value_projection_weight_initializer=None,
                 attention_value_projection_bias_initializer=None,
                 attention_kernel_initializer=None,
                 attention_query_projection_weight_initializer=None,
                 attention_out_projection_weight_initializer=None,
                 last_conv_kernel_initializer=None,
                 last_conv_bias_initializer=None,
                 done_weight_initializer=None,
                 done_bias_initializer=None,
                 training=False, trainable=True,
                 name=None, **kwargs):
        super(Decoder, self).__init__(name=name, trainable=trainable, **kwargs)
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.in_dim = in_dim
        self.r = r
        self.training = training
        self._collect_metrics = training

        self.query_position_rate = query_position_rate
        self.key_position_rate = key_position_rate

        self.mh_attentions = mh_attentions

        assert all([mha.out_channels == mh_attentions[0].out_channels for mha in mh_attentions])

        self.embed_query_positions = SinusoidalEncodingEmbedding(max_positions, mh_attentions[0].out_channels)
        self.embed_key_positions = SinusoidalEncodingEmbedding(max_positions, embed_dim)

        self.preattention = DecoderPreNet(in_channels=in_dim * r, params=preattention, dropout=dropout,
                                          is_incremental=is_incremental,
                                          kernel_initializer=prenet_weight_initializer,
                                          bias_initializer=prenet_bias_initializer,
                                          training=training)

        assert self.preattention.output_size == self.embed_query_positions.embedding_dim

        out_channels = mh_attentions[-1].out_channels
        self.last_conv = Conv1d(out_channels, in_dim * r, kernel_size=1, dilation=1, activation=None, dropout=dropout,
                                is_incremental=is_incremental, is_training=training,
                                kernel_initializer=last_conv_kernel_initializer,
                                bias_initializer=last_conv_bias_initializer, normalize_weight=True)

        self.fc = Linear(in_dim * r, 1, weight_initializer=done_weight_initializer,
                         bias_initializer=done_bias_initializer)

        self.max_decoder_steps = max_decoder_steps
        self.min_decoder_steps = min_decoder_steps
        self.use_memory_mask = use_memory_mask
        self.use_query_mask = use_query_mask

        self.is_incremental = is_incremental
        self.training = training

        self.attention_key_projection_weight_initializer = attention_key_projection_weight_initializer
        self.attention_key_projection_bias_initializer = attention_key_projection_bias_initializer
        self.attention_value_projection_weight_initializer = attention_value_projection_weight_initializer
        self.attention_value_projection_bias_initializer = attention_value_projection_bias_initializer

        self.attention_kernel_initializer = attention_kernel_initializer
        self.attention_query_projection_weight_initializer = attention_query_projection_weight_initializer
        self.attention_out_projection_weight_initializer = attention_out_projection_weight_initializer

    def build(self, input_shape):
        pass

    def call(self, encoder_out, input=None, text_positions=None, frame_positions=None, test_inputs=None,
             memory_mask=None, query_mask=None):
        if self.is_incremental:
            return self._call_incremental(encoder_out, text_positions, test_inputs)
        else:
            return self._call(encoder_out, input, text_positions=text_positions, frame_positions=frame_positions,
                              memory_mask=memory_mask, query_mask=query_mask)

    def _call(self, encoder_out, inputs, text_positions=None, frame_positions=None, memory_mask=None, query_mask=None):
        if inputs.shape[-1].value == self.in_dim:
            original_shape = inputs.shape
            s = tf.shape(inputs)
            new_shape = tf.stack([s[0], s[1] // self.r, tf.constant(-1, dtype=tf.int32)])
            inputs = tf.reshape(inputs, shape=new_shape)
            inputs.set_shape((original_shape[0].value, None, self.in_dim * self.r))

        keys, values = encoder_out

        if text_positions is not None:
            w = self.key_position_rate
            text_pos_embed = self.embed_key_positions(text_positions, w)
                
            keys = keys + text_pos_embed
            if self._collect_metrics:
                tf.summary.histogram("keys", keys)
                tf.summary.histogram("text_pos_embed", text_pos_embed)

        if frame_positions is not None:
            w = self.query_position_rate
            frame_pos_embed = self.embed_query_positions(frame_positions, w)
            if self._collect_metrics:
                tf.summary.histogram("frame_pos_embed", frame_pos_embed)
        else:
            raise ValueError("frame_positions is required")
            # frame_pos_embed = None

        x = inputs
        x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        x = self.preattention(x)

        attention_mechanism = ScaledDotProductAttentionMechanism(keys, values, self.embed_dim,
                                                                 key_projection_weight_initializer=self.attention_key_projection_weight_initializer,
                                                                 key_projection_bias_initializer=self.attention_key_projection_bias_initializer,
                                                                 value_projection_weight_initializer=self.attention_value_projection_weight_initializer,
                                                                 value_projection_bias_initializer=self.attention_value_projection_bias_initializer,
                                                                 training=self.training)
        memory_mask = memory_mask if self.use_memory_mask else None
        query_mask = query_mask if self.use_query_mask else None
        mp_attention = MultiHopAttention(attention_mechanism, self.preattention.output_size,
                                         self.mh_attentions, self.r, self.is_incremental,
                                         memory_mask=memory_mask,
                                         query_mask=query_mask,
                                         kernel_initializer=self.attention_kernel_initializer,
                                         query_projection_weight_initializer=self.attention_query_projection_weight_initializer,
                                         out_projection_weight_initializer=self.attention_out_projection_weight_initializer,
                                         training=self.training)

        x, alignments = mp_attention(CNNAttentionWrapperInput(x, frame_pos_embed),
                                     mp_attention.zero_state(inputs.shape[0].value,
                                                             inputs.dtype))  # ToDo: does not work when batch size is None

        if self._collect_metrics:
            mp_attention.register_metrics()

        x = self.last_conv(x.query)

        # project to mel-spectorgram
        outputs = tf.sigmoid(x)

        # Done flag
        # We need logits to compute loss with tf.losses.sigmoid_cross_entropy. No need to apply sigmoid here.
        done = self.fc(x)
        return outputs, done, alignments

    def _call_incremental(self, encoder_out, text_positions, test_inputs=None):
        keys, values = encoder_out
        batch_size = keys.shape[0].value
        # position encodings
        w = self.key_position_rate
        text_pos_embed = self.embed_key_positions(text_positions, w)
        keys = keys + text_pos_embed

        attention_mechanism = ScaledDotProductAttentionMechanism(keys, values, self.embed_dim,
                                                                 key_projection_weight_initializer=self.attention_key_projection_weight_initializer,
                                                                 key_projection_bias_initializer=self.attention_key_projection_bias_initializer,
                                                                 value_projection_weight_initializer=self.attention_value_projection_weight_initializer,
                                                                 value_projection_bias_initializer=self.attention_value_projection_bias_initializer,
                                                                 training=self.training)
        attention = MultiHopAttention(attention_mechanism, self.preattention.output_size,
                                      self.mh_attentions, self.r, self.is_incremental,
                                      kernel_initializer=self.attention_kernel_initializer,
                                      query_projection_weight_initializer=self.attention_query_projection_weight_initializer,
                                      out_projection_weight_initializer=self.attention_out_projection_weight_initializer,
                                      training=self.training)

        test_input_length = 0 if test_inputs is None else test_inputs.shape[1].value
        # append one element to avoid index overflow
        test_inputs = test_inputs if test_inputs is None else self.append_unused_final_test_input(test_inputs,
                                                                                                  batch_size)

        def condition(time, unused_input, unused_preattention_state, unused_attention_state, unused_frame_pos,
                      unused_last_conv_state,
                      unused_outputs, done):
            termination_criteria = tf.greater(done, 0.5)
            minimum_requirement = tf.greater(time, self.min_decoder_steps)
            maximum_criteria = tf.greater_equal(time, self.max_decoder_steps)
            termination = tf.logical_or(tf.logical_and(termination_criteria, minimum_requirement), maximum_criteria)
            # tf.while_loop continues body until cond returns False
            result = tf.logical_not(termination)
            return tf.squeeze(tf.reduce_all(result, axis=0))

        def test_condition(time, unused_input, unused_preattention_state, unused_attention_state, unused_frame_pos,
                           unused_last_conv_state,
                           unused_outputs, done):
            return tf.less(time, test_input_length)

        def body(time, input, preattention_state, attention_state, frame_pos, last_conv_state, outputs, done):
            w = self.query_position_rate
            frame_pos_embed = self.embed_query_positions(frame_pos, w)
            x = tf.layers.dropout(input, rate=self.dropout, training=self.training)
            x, next_preattention_state = self.preattention(x, state=preattention_state)
            (x, _), next_attention_states = attention.apply(CNNAttentionWrapperInput(x, frame_pos_embed),
                                                            attention_state)
            x, next_last_conv_state = self.last_conv(x, last_conv_state)
            # project to mel-spectorgram
            output = tf.sigmoid(x)
            # Done flag
            done = tf.squeeze(tf.sigmoid(self.fc(x)), axis=[1, 2])
            outputs = outputs.write(time, output)
            next_time = time + 1
            next_frame_pos = frame_pos + 1
            output = output if test_inputs is None else tf.expand_dims(test_inputs[:, next_time, :], axis=1)
            return (
                next_time, output, next_preattention_state, next_attention_states, next_frame_pos, next_last_conv_state,
                outputs,
                done)

        time = tf.constant(0)
        outputs_ta = tf.TensorArray(dtype=tf.float32, size=self.max_decoder_steps,
                                    element_shape=tf.TensorShape([batch_size, 1, self.in_dim * self.r]))
        initial_done = tf.constant(shape=[batch_size], value=0, dtype=tf.float32)
        condition_function = condition if test_inputs is None else test_condition
        initial_input = self.initial_input(batch_size) if test_inputs is None else tf.expand_dims(test_inputs[:, 0, :],
                                                                                                  axis=1)
        _, _, _, final_attention_state, _, _, out_online_ta, _ = tf.while_loop(condition_function, body, (
            time, initial_input, self.preattention.zero_state(batch_size, tf.float32),
            attention.zero_state(batch_size, tf.float32),
            self.initial_frame_pos(batch_size), self.last_conv.zero_state(batch_size, tf.float32), outputs_ta,
            initial_done))
        output_online = nest.map_structure(lambda ta: ta.stack(), out_online_ta)

        output_online = tf.squeeze(output_online, axis=2)
        output_online = tf.transpose(output_online, perm=(1, 0, 2))

        return output_online

    def initial_input(self, batch_size):
        return tf.zeros(shape=(batch_size, 1, self.in_dim * self.r))

    def initial_frame_pos(self, batch_size):
        # frame pos start with 1.
        time = 1
        return tf.fill(dims=(batch_size, 1), value=time)

    def append_unused_final_test_input(self, test_input, batch_size):
        final_input = tf.zeros(shape=(batch_size, 1, self.in_dim * self.r))
        return tf.concat([test_input, final_input], axis=1)

    def register_metrics(self):
        self.embed_key_positions.register_metrics()
        self.embed_query_positions.register_metrics()
        self.preattention.register_metrics()
        self.fc.register_metrics()
        self.last_conv.register_metrics()
