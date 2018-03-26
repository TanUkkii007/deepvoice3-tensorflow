import tensorflow as tf
import collections
from data import PreprocessedTargetData, PreprocessedSourceData
from data.tfrecord_utils import parse_preprocessed_source_data, parse_preprocessed_target_data, \
    decode_preprocessed_source_data, decode_preprocessed_target_data


class PreparedSourceData(collections.namedtuple("PreparedSourceData",
                                                ["id", "text", "source", "source_length", "text_positions",
                                                 "text2", "source2", "source_length2", "text_positions2"])):
    pass


class _PreparedTargetData(
    collections.namedtuple("PreparedTargetData",
                           ["id", "spec", "spec_width", "mel", "mel_width", "target_length", "done"])):
    pass


class PreparedTargetData(
    collections.namedtuple("PreparedTargetData",
                           ["id", "spec", "spec_width", "mel", "mel_width", "target_length", "done",
                            "frame_positions"])):
    pass


class Frontend():

    def __init__(self, source, target, hparams):
        self.source = source
        self.target = target
        self.hparams = hparams

    def _decode_source(self):
        return self.source.map(lambda d: decode_preprocessed_source_data(parse_preprocessed_source_data(d)))

    def _decode_target(self):
        return self.target.map(lambda d: decode_preprocessed_target_data(parse_preprocessed_target_data(d)))

    def prepare_source(self):
        def convert(inputs: PreprocessedSourceData):
            input_length = inputs.source_length
            input_length2 = inputs.source_length2
            # text position
            text_positions1 = tf.range(1, input_length + 1)
            text_positions2 = tf.range(1, input_length2 + 1)

            return PreparedSourceData(inputs.id, inputs.text, inputs.source, inputs.source_length, text_positions1,
                                      inputs.text2, inputs.source2, inputs.source_length2, text_positions2)

        return self._decode_source().map(lambda inputs: convert(inputs))

    def prepare_target(self):
        def convert(target: PreprocessedTargetData):
            r = self.hparams.outputs_per_step
            downsample_step = self.hparams.downsample_step

            # Set 0 for zero beginning padding
            # imitates initial decoder states
            b_pad = r
            spec = tf.pad(target.spec, paddings=tf.constant([[b_pad, 0], [0, 0]]))
            mel = tf.pad(target.mel, paddings=tf.constant([[b_pad, 0], [0, 0]]))
            target_length = target.target_length + b_pad

            # done flag
            done = tf.zeros(target_length // r // downsample_step - 1, dtype=tf.float32)
            return _PreparedTargetData(target.id, spec, target.spec_width, mel, target.mel_width, target_length, done)

        return self._decode_target().map(lambda inputs: convert(inputs))

    @staticmethod
    def zip_source_and_target(source: tf.data.Dataset, target: tf.data.Dataset):
        def assert_id(source, target):
            with tf.control_dependencies([tf.assert_equal(source.id, target.id)]):
                return (source, target)

        return tf.data.Dataset.zip((source, target)).map(lambda x, y: assert_id(x, y))

    @staticmethod
    def batch_dataset(dataset: tf.data.Dataset, batch_size):
        def key_func(source, target):
            bucket_width = 10
            num_buckets = 100
            bucket_id = target.target_length // bucket_width
            return tf.to_int64(tf.minimum(num_buckets, bucket_id))

        def reduce_func(window: tf.data.Dataset):
            return window.padded_batch(batch_size, padded_shapes=(
                PreparedSourceData(
                    id=tf.TensorShape([]),
                    source=tf.TensorShape([None]),
                    source_length=tf.TensorShape([]),
                    text_positions=tf.TensorShape([None]),
                    source2=tf.TensorShape([None]),
                    source_length2=tf.TensorShape([]),
                    text_positions2=tf.TensorShape([None]),
                ),
                _PreparedTargetData(
                    id=tf.TensorShape([]),
                    spec=tf.TensorShape([None]),
                    spec_width=tf.TensorShape([]),
                    mel=tf.TensorShape([None]),
                    mel_width=tf.TensorShape([]),
                    target_length=tf.TensorShape([]),
                    done=tf.TensorShape([None]),
                )), padding_values=(
                PreparedSourceData(
                    id=0,
                    source=0,
                    source_length=0,
                    text_positions=0,
                    source2=0,
                    source_length2=0,
                    text_positions2=0,
                ),
                _PreparedTargetData(
                    id=0,
                    spec=0,
                    spec_width=0,
                    mel=0,
                    mel_width=0,
                    target_length=0,
                    done=1,
                )))

        return dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                             reduce_func,
                                                             window_size=batch_size))


