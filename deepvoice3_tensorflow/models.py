import tensorflow as tf
from deepvoice3_tensorflow.deepvoice3 import Encoder, Decoder, Converter, DecoderPreNetArgs, MultiHopAttentionArgs
from deepvoice3_tensorflow.hooks import AlignmentSaver, ConverterMetricsSaver


class SingleSpeakerTTSModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):

        def model_fn(features, labels, mode, params):
            is_incremental = mode is not tf.estimator.ModeKeys.TRAIN
            training = mode == tf.estimator.ModeKeys.TRAIN

            dropout = params.dropout
            k = params.kernel_size
            eh = params.encoder_channels
            dh = params.decoder_channels

            preattention = [DecoderPreNetArgs(dh // 2), DecoderPreNetArgs(dh)]
            mhattention = [MultiHopAttentionArgs(dh, k, 1, 0.0),
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

            global_step = tf.train.get_global_step()

            if training:
                r = params.outputs_per_step
                mel_outputs, done_hat, attention_states = decoder((keys, values), input=labels.mel,
                                                                  frame_positions=labels.frame_positions,
                                                                  text_positions=features.text_positions,
                                                                  memory_mask=features.mask)
                # undo reduction
                mel_outputs = tf.reshape(mel_outputs, shape=(params.batch_size, -1, params.num_mels))

                alignments = [s.alignments for s in attention_states]
                # drop last unused frame and artificial initial zero frame
                mel_loss = spec_loss(mel_outputs[:, :-r, :], labels.mel[:, r:, :], labels.spec_loss_mask[:, r:])
                done_loss = binary_loss(done_hat, labels.done, labels.binary_loss_mask)
                loss = mel_loss + done_loss
                optimizer = tf.train.AdamOptimizer(learning_rate=params.initial_learning_rate, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)
                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = AlignmentSaver(alignments, global_step, mel_outputs, labels.mel, features.id,
                                                 features.text,
                                                 params.alignment_save_steps,
                                                 "alignment_layer", mode, summary_writer)
                add_stats(encoder, decoder, mel_loss, done_loss)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[alignment_saver])

            if mode == tf.estimator.ModeKeys.EVAL:
                test_inputs = labels.mel if params.teacher_forcing else None
                mel_outputs, done_hat, attention_states = decoder((keys, values),
                                                                  text_positions=features.text_positions,
                                                                  test_inputs=test_inputs)
                # undo reduction
                mel_outputs = tf.reshape(mel_outputs, shape=(params.batch_size, -1, params.num_mels))
                alignments = [tf.transpose(tf.squeeze(s.alignment_history.stack(), axis=2), perm=[1, 0, 2]) for s in
                              attention_states]

                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = AlignmentSaver(alignments, global_step, mel_outputs, labels.mel, features.id,
                                                 features.text,
                                                 save_steps=1,
                                                 tag_prefix="eval_alignment_layer", mode=mode, writer=summary_writer)
                # ToDo: calculate loss
                return tf.estimator.EstimatorSpec(mode, loss=tf.constant(0),
                                                  evaluation_hooks=[alignment_saver])

        def spec_loss(y_hat, y, mask, priority_bin=None, priority_w=0):
            l1_loss = tf.abs(y_hat - y)

            # Priority L1 loss
            if priority_bin is not None and priority_w > 0:
                priority_loss = tf.abs(y_hat[:, :, :priority_bin] - y[:, :, :priority_bin])
                l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

            return tf.losses.compute_weighted_loss(l1_loss, weights=tf.expand_dims(mask, axis=2))

        def binary_loss(done_hat, done, mask):
            return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1), weights=mask)

        def add_stats(encoder, decoder, mel_loss, done_loss):
            tf.summary.scalar("mel_loss", mel_loss)
            tf.summary.scalar("done_loss", done_loss)
            encoder.register_metrics()
            decoder.register_metrics()
            return tf.summary.merge_all()

        super(SingleSpeakerTTSModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)


class DeepVoice3ConverterModel(tf.estimator.Estimator):

    def __init__(self, params, model_dir=None, config=None, warm_start_from=None):
        def model_fn(features, labels, mode, params):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            is_validation = mode == tf.estimator.ModeKeys.EVAL

            time_upsampling = max(params.downsample_step // params.outputs_per_step, 1)
            c = params.converter_channels
            k = params.converter_kernel_size
            n = params.converter_layers

            converter = Converter(in_dim=params.num_mels,
                                  out_dim=params.fft_size // 2 + 1,
                                  convolutions=((c, k, 1),) * n,
                                  time_upsampling=time_upsampling,
                                  dropout=params.dropout)

            linear_output = converter(features.mel)

            global_step = tf.train.get_global_step()

            if mode is not tf.estimator.ModeKeys.PREDICT:
                linear_loss = self.spec_loss(linear_output, labels.spec,
                                             labels.spec_loss_mask)
                loss = linear_loss

            if is_training:
                lr = self.learning_rate_decay(
                    params.initial_learning_rate, global_step) if params.decay_learning_rate else tf.convert_to_tensor(
                    params.initial_learning_rate)
                optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)

                gradients, variables = zip(*optimizer.compute_gradients(loss))
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                self.add_training_stats(linear_loss, lr)
                summary_writer = tf.summary.FileWriter(model_dir)
                metrics_saver = ConverterMetricsSaver(global_step, linear_output, labels.spec,
                                                      labels.target_length,
                                                      features.id,
                                                      params.alignment_save_steps,
                                                      mode, summary_writer)

                train_op = optimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                  training_hooks=[metrics_saver])

            if is_validation:
                eval_metric_ops = self.get_validation_metrics(linear_loss)
                summary_writer = tf.summary.FileWriter(model_dir)
                metrics_saver = ConverterMetricsSaver(global_step, linear_output, labels.spec,
                                                      labels.target_length,
                                                      features.id,
                                                      1,
                                                      mode, summary_writer)
                return tf.estimator.EstimatorSpec(mode, loss=loss,
                                                  eval_metric_ops=eval_metric_ops,
                                                  evaluation_hooks=[metrics_saver])

        super(DeepVoice3ConverterModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)

    @staticmethod
    def spec_loss(y_hat, y, mask, n_priority_freq=None, priority_w=0):
        l1_loss = tf.abs(y_hat - y)

        # Priority L1 loss
        if n_priority_freq is not None and priority_w > 0:
            priority_loss = tf.abs(y_hat[:, :, :n_priority_freq] - y[:, :, :n_priority_freq])
            l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

        return tf.losses.compute_weighted_loss(l1_loss, weights=tf.expand_dims(mask, axis=2))

    @staticmethod
    def learning_rate_decay(init_rate, global_step):
        warmup_steps = 4000.0
        step = tf.to_float(global_step + 1)
        return init_rate * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

    @staticmethod
    def add_training_stats(linear_loss, learning_rate):
        if linear_loss is not None:
            tf.summary.scalar("linear_loss", linear_loss)
        tf.summary.scalar("learning_rate", learning_rate)
        return tf.summary.merge_all()

    @staticmethod
    def get_validation_metrics(linear_loss):
        metrics = {}
        if linear_loss is not None:
            metrics["linear_loss"] = tf.metrics.mean(linear_loss)
        return metrics
