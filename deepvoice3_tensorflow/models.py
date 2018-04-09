import tensorflow as tf
from deepvoice3_tensorflow.deepvoice3 import Encoder, Decoder, DecoderPreNetCNNArgs, MultiHopAttentionArgs
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os


def save_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


class AlignmentSaver(tf.train.SessionRunHook):

    def __init__(self, alignment_tensors, global_step_tensor, save_steps, tag_prefix, writer: tf.summary.FileWriter):
        self.alignment_tensors = alignment_tensors
        self.global_step_tensor = global_step_tensor
        self.save_steps = save_steps
        self.tag_prefix = tag_prefix
        self.writer = writer
        self.last_step = None

    def before_run(self, run_context):
        return tf.train.SessionRunArgs({
            "global_step": self.global_step_tensor,
            "alignments": self.alignment_tensors
        })

    def after_run(self,
                  run_context,
                  run_values):
        stale_global_step = run_values.results["global_step"]
        alignments = run_values.results["alignments"]
        if (stale_global_step + 1) % self.save_steps == 0 or stale_global_step == 0:
            global_step_value = run_context.session.run(self.global_step_tensor)
            for i, alignment in enumerate(alignments):
                sample_idx = 0
                sample_alignment = alignment[sample_idx]
                tag = self.tag_prefix + str(i)
                file_path = os.path.join(self.writer.get_logdir(), tag + "_step{:09d}.png".format(global_step_value))
                tf.logging.info("Saving alignments for %d at %s", global_step_value, file_path)
                save_alignment(sample_alignment.T, file_path, info="global_step={}".format(global_step_value))


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
            mel_outputs, done_hat, attention_states = decoder((keys, values), input=labels.mel,
                                                              frame_positions=labels.frame_positions,
                                                              text_positions=features.text_positions)

            # undo reduction
            mel_outputs = tf.reshape(mel_outputs, shape=(params.batch_size, -1, params.num_mels))

            if training:
                alignments = [s.alignments for s in attention_states]
                mel_loss = spec_loss(mel_outputs, labels.mel)
                done_loss = binary_loss(done_hat, labels.done)
                loss = mel_loss + done_loss
                optimizer = tf.train.AdamOptimizer(learning_rate=params.initial_learning_rate, beta1=params.adam_beta1,
                                                   beta2=params.adam_beta2, epsilon=params.adam_eps)
                global_step = tf.train.get_global_step()
                train_op = optimizer.minimize(loss, global_step=global_step)
                summary_writer = tf.summary.FileWriter(model_dir)
                alignment_saver = AlignmentSaver(alignments, global_step, params.alignment_save_steps,
                                                 "alignment_layer", summary_writer)
                add_stats(encoder, decoder, mel_loss, done_loss)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[alignment_saver])

        def spec_loss(y_hat, y, priority_bin=None, priority_w=0):
            l1_loss = tf.abs(y_hat - y)

            # Priority L1 loss
            if priority_bin is not None and priority_w > 0:
                priority_loss = tf.abs(y_hat[:, :, :priority_bin] - y[:, :, :priority_bin])
                l1_loss = (1 - priority_w) * l1_loss + priority_w * priority_loss

            return tf.losses.compute_weighted_loss(l1_loss)

        def binary_loss(done_hat, done):
            return tf.losses.sigmoid_cross_entropy(done, tf.squeeze(done_hat, axis=-1))

        def add_stats(encoder, decoder, mel_loss, done_loss):
            tf.summary.scalar("mel_loss", mel_loss)
            tf.summary.scalar("done_loss", done_loss)
            encoder.register_metrics()
            decoder.register_metrics()
            return tf.summary.merge_all()

        super(SingleSpeakerTTSModel, self).__init__(
            model_fn=model_fn, model_dir=model_dir, config=config,
            params=params, warm_start_from=warm_start_from)
