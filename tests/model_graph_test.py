import tensorflow as tf
import os
import tempfile
from deepvoice3_tensorflow.models import SingleSpeakerTTSModel
from deepvoice3_tensorflow.frontend import Frontend


class ModelTest(tf.test.TestCase):

    def test_train(self):
        tf.logging.set_verbosity(tf.logging.INFO)
        model_dir = tempfile.mkdtemp()
        r = 1
        hparams = tf.contrib.training.HParams(
            num_mels=80,
            fft_size=1024,
            downsample_step=4,
            outputs_per_step=r,

            max_positions=512,
            n_vocab=0xffff,
            dropout=1 - 0.95,
            kernel_size=3,
            text_embed_dim=128,
            text_embedding_weight_std=0.1,
            encoder_channels=256,
            decoder_channels=256,
            max_decoder_steps=200,
            min_decoder_steps=10,
            query_position_rate=1.0,
            key_position_rate=2.37,
            key_projection=False,
            value_projection=False,
            use_memory_mask=True,

            batch_size=2,
            approx_min_target_length=200,
            batch_bucket_width=50,
            batch_num_buckets=3,
            initial_learning_rate=5e-4,
            adam_beta1=0.5,
            adam_beta2=0.9,
            adam_eps=1e-6,
            alignment_save_steps=2,
        )

        def train_input_fn():
            data_dir = os.path.join(os.path.dirname(__file__), "test_data")
            source_files = [os.path.join(data_dir, "jsut-source-%05d.tfrecords" % i) for i in range(1, 11)]
            target_files = [os.path.join(data_dir, "jsut-target-%05d.tfrecords" % i) for i in range(1, 11)]
            source = tf.data.TFRecordDataset(source_files)
            target = tf.data.TFRecordDataset(target_files)

            frontend = Frontend(source, target, hparams)
            batched = frontend.prepare().zip_source_and_target().group_by_batch().add_frame_positions().downsample_mel().dataset
            return batched

        estimator = SingleSpeakerTTSModel(hparams, model_dir)

        estimator.train(lambda: train_input_fn(), steps=5)
