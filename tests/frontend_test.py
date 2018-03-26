import tensorflow as tf
import numpy as np
import os
from deepvoice3_tensorflow.frontend import Frontend


class FrontendTest(tf.test.TestCase):

    def test_preparation(self):
        data_dir = "test_data"
        source_files = [os.path.join(data_dir, "jsut-source-%05d.tfrecords" % i) for i in range(1, 11)]
        target_files = [os.path.join(data_dir, "jsut-target-%05d.tfrecords" % i) for i in range(1, 11)]
        source = tf.data.TFRecordDataset(source_files)
        target = tf.data.TFRecordDataset(target_files)

        hparams = tf.contrib.training.HParams(
            downsample_step=4,
            outputs_per_step=1,
            batch_size=2,
            approx_min_target_length=200,
            batch_bucket_width=50,
            batch_num_buckets=3,
        )

        frontend = Frontend(source, target, hparams)

        zipped = frontend.prepare().zip_source_and_target().dataset

        with self.test_session() as sess:
            iterator = zipped.make_one_shot_iterator()
            next_element = iterator.get_next()
            for _ in range(10):
                s, t = sess.run(next_element)
                self.assertEqual(s.id, t.id)
                self.assertEqual(s.source_length, len(s.source))
                self.assertEqual(s.source_length2, len(s.source2))
                # drop EOS
                self.assertEqual([ord(c) for c in s.text.decode('utf-8')], list(s.source[:-1]))
                self.assertEqual([ord(c) for c in s.text2.decode('utf-8')], list(s.source2[:-1]))

                self.assertEqual(t.target_length, len(t.spec))
                self.assertEqual(t.target_length, len(t.mel))

    def test_batch(self):
        data_dir = "test_data"
        source_files = [os.path.join(data_dir, "jsut-source-%05d.tfrecords" % i) for i in range(1, 11)]
        target_files = [os.path.join(data_dir, "jsut-target-%05d.tfrecords" % i) for i in range(1, 11)]
        source = tf.data.TFRecordDataset(source_files)
        target = tf.data.TFRecordDataset(target_files)

        r = 1

        hparams = tf.contrib.training.HParams(
            num_mels=80,
            downsample_step=4,
            outputs_per_step=r,
            batch_size=2,
            approx_min_target_length=200,
            batch_bucket_width=50,
            batch_num_buckets=3,
        )

        frontend = Frontend(source, target, hparams)

        batched = frontend.prepare().zip_source_and_target().group_by_batch().dataset

        with self.test_session() as sess:
            iterator = batched.make_one_shot_iterator()
            next_element = iterator.get_next()
            for _ in range(5):
                s, t = sess.run(next_element)

                source_length = s.source_length
                source_length2 = s.source_length2
                max_source_length = len(s.source[0])
                max_source_length2 = len(s.source2[0])
                # source1 padding
                self.assertAllEqual(np.zeros(max_source_length - source_length[0]), s.source[0][source_length[0]:])
                self.assertAllEqual(np.zeros(max_source_length - source_length[1]), s.source[1][source_length[1]:])
                # source2 padding
                self.assertAllEqual(np.zeros(max_source_length2 - source_length2[0]), s.source2[0][source_length2[0]:])
                self.assertAllEqual(np.zeros(max_source_length2 - source_length2[1]), s.source2[1][source_length2[1]:])

                # text_positions1 padding
                self.assertAllEqual(np.zeros(max_source_length - source_length[0]),
                                    s.text_positions[0][source_length[0]:])
                self.assertAllEqual(np.zeros(max_source_length - source_length[1]),
                                    s.text_positions[1][source_length[1]:])
                # text_positions2 padding
                self.assertAllEqual(np.zeros(max_source_length2 - source_length2[0]),
                                    s.text_positions2[0][source_length2[0]:])
                self.assertAllEqual(np.zeros(max_source_length2 - source_length2[1]),
                                    s.text_positions2[1][source_length2[1]:])

                target_length1 = t.target_length[0]
                target_length2 = t.target_length[1]
                max_target_length = len(t.mel[0])

                # mel padding
                self.assertAllEqual(np.zeros([max_target_length - target_length1, hparams.num_mels]),
                                    t.mel[0][target_length1:])
                self.assertAllEqual(np.zeros([max_target_length - target_length2, hparams.num_mels]),
                                    t.mel[1][target_length2:])

                # spec padding
                self.assertAllEqual(np.zeros([max_target_length - target_length1, 513]),
                                    t.spec[0][target_length1:])
                self.assertAllEqual(np.zeros([max_target_length - target_length2, 513]),
                                    t.spec[1][target_length2:])

                # done
                self.assertAllEqual(np.zeros(target_length1 // r // hparams.downsample_step - 1),
                                    t.done[0][:target_length1 // r // hparams.downsample_step - 1])
                self.assertAllEqual(np.zeros(target_length2 // r // hparams.downsample_step - 1),
                                    t.done[1][:target_length2 // r // hparams.downsample_step - 1])
                self.assertAllEqual(np.ones((max_target_length // r // hparams.downsample_step) - (
                            target_length1 // r // hparams.downsample_step - 1)),
                                    t.done[0][target_length1 // r // hparams.downsample_step - 1:])
                self.assertAllEqual(np.ones((max_target_length // r // hparams.downsample_step) - (
                        target_length2 // r // hparams.downsample_step - 1)),
                                    t.done[1][target_length2 // r // hparams.downsample_step - 1:])
