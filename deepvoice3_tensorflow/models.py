import tensorflow as tf
from deepvoice3_tensorflow.deepvoice3 import Encoder, Decoder, DecoderPreNetCNNArgs, MultiHopAttentionArgs


class SingleSpeakerTTSModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):

        def model_fn(features, labels, mode, params):
            is_incremental = mode is not tf.estimator.ModeKeys.TRAIN
            training = mode == tf.estimator.ModeKeys.TRAIN

            dropout = params.dropout
            k = params.kernel_size
            eh = params.encoder_channels
            dh = params.decoder_channels

            preattention = [DecoderPreNetCNNArgs(dh, k, 1), DecoderPreNetCNNArgs(dh, k, 3)]
            mhattention = [MultiHopAttentionArgs(dh, k, 1, dropout),
                           MultiHopAttentionArgs(dh, k, 3, dropout),
                           MultiHopAttentionArgs(dh, k, 9, dropout),
                           MultiHopAttentionArgs(dh, k, 27, dropout),
                           MultiHopAttentionArgs(dh, k, 1, dropout), ]

            encoder = Encoder(params.n_vocab, params.text_embed_dim, params.text_embedding_weight_std,
                              convolutions=[(eh, k, 1), (eh, k, 3), (eh, k, 9), (eh, k, 27),
                                            (eh, k, 1), (eh, k, 3), (eh, k, 9), (eh, k, 27),
                                            (eh, k, 1), (eh, k, 3)],
                              dropout=dropout,
                              training=training)

            decoder = Decoder(params.text_embed_dim, params.num_mels, params.outputs_per_step, params.max_positions,
                              preattention=preattention,
                              mh_attentions=mhattention,
                              dropout=dropout,
                              use_memory_mask=params.use_memory_mask,
                              query_position_rate=params.query_position_rate,
                              key_position_rate=params.key_position_rate,
                              max_decoder_steps=params.max_decoder_steps,
                              min_decoder_steps=params.min_decoder_steps,
                              is_incremental=is_incremental,
                              training=training)

            keys, values = encoder(features.source, text_positions=features.text_positions)
            mel_outputs, done_hat = decoder((keys, values), input=labels.mel,
                                            frame_positions=labels.frame_positions,
                                            text_positions=features.text_positions)

            # undo reduction
            mel_outputs = tf.reshape(mel_outputs, shape=(params.batch_size, -1, params.num_mels))

            if training:
                mel_loss = spec_loss(mel_outputs, labels.mel)
                done_loss = binary_loss(done_hat, labels.done)
                loss = mel_loss + done_loss
                optimizer = tf.train.AdamOptimizer(learning_rate=params.initial_learning_rate, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)
                train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        def spec_loss(y_hat, y, priority_bin=None, priority_w=0):
            l1_loss = tf.abs(y_hat - y)

            # Priority L1 loss
            if priority_bin is not None and priority_w > 0:
                priority_loss = tf.abs(y_hat[:, :, :priority_bin] - y[:, :, :priority_bin])
                l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

            return tf.losses.compute_weighted_loss(l1_loss)

        def binary_loss(done_hat, done):
            return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1))

        super(SingleSpeakerTTSModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)
