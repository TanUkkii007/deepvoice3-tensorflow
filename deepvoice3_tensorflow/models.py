import tensorflow as tf
from deepvoice3_tensorflow.deepvoice3 import Encoder, Decoder, DecoderPreNetArgs, MultiHopAttentionArgs
import os
import numpy as np
from typing import List
from data.tfrecord_utils import write_tfrecord, int64_feature, bytes_feature


def write_training_result(global_step: int, id: List[int], text: List[str], predicted_mel: List[np.ndarray],
                          ground_truth_mel: List[np.ndarray], alignment: List[np.ndarray], filename: str):
    batch_size = len(ground_truth_mel)
    raw_predicted_mel = [m.tostring() for m in predicted_mel]
    raw_ground_truth_mel = [m.tostring() for m in ground_truth_mel]
    mel_width = ground_truth_mel[0].shape[1]
    mel_length = [m.shape[0] for m in ground_truth_mel]
    raw_alignment = [a.tostring() for a in alignment]
    alignment_source_length = [a.shape[1] for a in alignment]
    alignment_target_length = [a.shape[2] for a in alignment]
    example = tf.train.Example(features=tf.train.Features(feature={
        'global_step': int64_feature([global_step]),
        'batch_size': int64_feature([batch_size]),
        'id': int64_feature(id),
        'text': bytes_feature(text),
        'predicted_mel': bytes_feature(raw_predicted_mel),
        'ground_truth_mel': bytes_feature(raw_ground_truth_mel),
        'mel_length': int64_feature(mel_length),
        'mel_width': int64_feature([mel_width]),
        'alignment': bytes_feature(raw_alignment),
        'alignment_source_length': int64_feature(alignment_source_length),
        'alignment_target_length': int64_feature(alignment_target_length),
    }))
    write_tfrecord(example, filename)


class AlignmentSaver(tf.train.SessionRunHook):

    def __init__(self, alignment_tensors, global_step_tensor, predicted_mel_tensor, ground_truth_mel_tensor, id_tensor,
                 text_tensor, save_steps,
                 tag_prefix, mode, writer: tf.summary.FileWriter):
        self.alignment_tensors = alignment_tensors
        self.global_step_tensor = global_step_tensor
        self.predicted_mel_tensor = predicted_mel_tensor
        self.ground_truth_mel_tensor = ground_truth_mel_tensor
        self.id_tensor = id_tensor
        self.text_tensor = text_tensor
        self.save_steps = save_steps
        self.tag_prefix = tag_prefix
        self.mode = mode
        self.writer = writer

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value, alignments, predicted_mel, ground_truth_mel, id, text = run_context.session.run(
                (self.global_step_tensor, self.alignment_tensors, self.predicted_mel_tensor,
                 self.ground_truth_mel_tensor, self.id_tensor, self.text_tensor))
            result_filename = "{}_result_step{:09d}.tfrecord".format(self.mode, global_step_value)
            tf.logging.info("Saving a training result for %d at %s", global_step_value, result_filename)
            write_training_result(global_step_value, list(id), list(text), list(predicted_mel), list(ground_truth_mel),
                                  alignments,
                                  filename=os.path.join(self.writer.get_logdir(), result_filename))


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
                mel_outputs, done_hat, attention_states = decoder((keys, values), input=labels.mel,
                                                                  frame_positions=labels.frame_positions,
                                                                  text_positions=features.text_positions,
                                                                  memory_mask=features.mask)
                # undo reduction
                mel_outputs = tf.reshape(mel_outputs, shape=(params.batch_size, -1, params.num_mels))

                alignments = [s.alignments for s in attention_states]
                mel_loss = spec_loss(mel_outputs, labels.mel, labels.spec_loss_mask)
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
                mel_outputs, done_hat, attention_states = decoder((keys, values),
                                                                  text_positions=features.text_positions)
                # undo reduction
                mel_outputs = tf.reshape(mel_outputs, shape=(params.batch_size, -1, params.num_mels))
                alignments = [tf.transpose(tf.squeeze(s.alignment_history.stack(), axis=2), perm=[1, 0, 2]) for s in
                              attention_states]

                # mel_loss = spec_loss(mel_outputs[:, tf.shape(labels.mel)[1], :], labels.mel, labels.spec_loss_mask)
                # done_loss = binary_loss(done_hat, labels.done, labels.binary_loss_mask)
                # loss = mel_loss + done_loss
                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = AlignmentSaver(alignments, global_step, mel_outputs, labels.mel, features.id,
                                                 features.text,
                                                 save_steps=1,
                                                 tag_prefix="eval_alignment_layer", mode=mode, writer=summary_writer)
                # ToDo: calculate loss
                return tf.estimator.EstimatorSpec(mode, loss=tf.constant(0), evaluation_hooks=[alignment_saver])

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
