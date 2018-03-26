import tensorflow as tf
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
        )

        frontend = Frontend(source, target, hparams)

        zipped = Frontend.zip_source_and_target(frontend.prepare_source(), frontend.prepare_target())

        with self.test_session() as sess:
            iterator = zipped.make_one_shot_iterator()
            next_element = iterator.get_next()
            for _ in range(10):
                s, t = sess.run(next_element)
                self.assertEqual(s.id, t.id)
                self.assertEqual(s.source_length, len(s.source))
                self.assertEqual(s.source_length2, len(s.source2))
                # drop EOS
                self.assertEqual([ord(c)  for c in s.text.decode('utf-8')], list(s.source[:-1]))
                self.assertEqual([ord(c)  for c in s.text2.decode('utf-8')], list(s.source2[:-1]))

                self.assertEqual(t.target_length, len(t.spec))
                self.assertEqual(t.target_length, len(t.mel))


