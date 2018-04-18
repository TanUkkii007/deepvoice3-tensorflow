import tensorflow as tf
import collections
import math
from abc import abstractmethod
from data import PreprocessedTargetData, PreprocessedSourceData
from data.tfrecord_utils import parse_preprocessed_source_data, parse_preprocessed_target_data, \
    decode_preprocessed_source_data, decode_preprocessed_target_data


class PreparedSourceData(collections.namedtuple("PreparedSourceData",
                                                ["id", "text", "source", "source_length", "text_positions",
                                                 "text2", "source2", "source_length2", "text_positions2"])):
    pass

class PreparedSourceDataWithMask(collections.namedtuple("PreparedSourceData",
                                                ["id", "text", "source", "source_length", "text_positions", "mask",
                                                 "text2", "source2", "source_length2", "text_positions2", "mask2"])):
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

class PreparedTargetDataWithMask(
    collections.namedtuple("PreparedTargetDataWithMask",
                           ["id", "spec", "spec_width", "mel", "mel_width", "target_length", "done",
                            "frame_positions", "spec_loss_mask", "binary_loss_mask"])):
    pass


def _lcm(a, b):
    return a * b // math.gcd(a, b)


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

            # spec and mel length must be multiple of outputs_per_step and downsample_step
            length_factor = _lcm(self.hparams.outputs_per_step, self.hparams.downsample_step)

            def padding_function(t):
                padded_target_length = (target_length // length_factor + 1) * length_factor
                tail_padding = padded_target_length - target_length
                padding_shape = tf.sparse_tensor_to_dense(
                    tf.SparseTensor(indices=[(0, 1)], values=tf.expand_dims(tail_padding, axis=0), dense_shape=(2, 2)))
                return lambda: tf.pad(t, paddings=padding_shape)

            no_padding_condition = tf.equal(tf.to_int64(0), target_length % length_factor)

            spec = tf.cond(no_padding_condition, lambda: spec, padding_function(spec))
            mel = tf.cond(no_padding_condition, lambda: mel, padding_function(mel))

            spec.set_shape((None, self.hparams.fft_size // 2 + 1))
            mel.set_shape((None, self.hparams.num_mels))

            # done flag
            # if padding is needed, done length should be +1
            done_tail_size = tf.cond(no_padding_condition, lambda: 1, lambda: 2)
            done = tf.concat([tf.zeros(target_length // r // downsample_step - 1, dtype=tf.float32),
                              tf.ones(done_tail_size, dtype=tf.float32)], axis=0)
            return _PreparedTargetData(target.id, spec, target.spec_width, mel, target.mel_width, target_length, done)

        return self._decode_target().map(lambda inputs: convert(inputs))

    def prepare(self):
        return _FrontendPreparedView(self.prepare_source(), self.prepare_target(), self.hparams)


class _FrontendPreparedView():
    def __init__(self, source: tf.data.Dataset, target: tf.data.Dataset, hparams):
        self.source = source
        self.target = target
        self.hparams = hparams

    def zip_source_and_target(self):
        def assert_id(source, target):
            with tf.control_dependencies([tf.assert_equal(source.id, target.id)]):
                return (source, target)

        zipped = tf.data.Dataset.zip((self.source, self.target)).map(lambda x, y: assert_id(x, y))
        return _FrontendZippedView(zipped, self.hparams)


class FrontendZippedViewBase:

    @property
    @abstractmethod
    def dataset(self):
        raise NotImplementedError("dataset")

    @property
    @abstractmethod
    def hparams(self):
        raise NotImplementedError("hparams")

    @abstractmethod
    def apply(self, dataset, hparams):
        raise NotImplementedError("apply")

    def shuffle(self, buffer_size):
        return self.apply(self.dataset.shuffle(buffer_size), self.hparams)

    def repeat(self, count=None):
        return self.apply(self.dataset.repeat(count), self.hparams)

    def swap_source_random(self, swap_probability):
        def convert(s: PreparedSourceData, t):
            r = tf.random_uniform(shape=(), minval=0, maxval=1)
            def s1():
                return s.text, s.source, s.source_length, s.text_positions
            def s2():
                return s.text2, s.source2, s.source_length2, s.text_positions2
            condition = r > swap_probability
            text, source, source_length, text_positions = tf.cond(condition, s1, s2)
            text2, source2, source_length2, text_positions2 = tf.cond(condition, s2, s1)

            return PreparedSourceData(
                id=s.id,
                text=text,
                source=source,
                source_length=source_length,
                text_positions=text_positions,
                text2=text2,
                source2=source2,
                source_length2=source_length2,
                text_positions2=text_positions2,
            ), t

        return self.apply(self.dataset.map(lambda x, y: convert(x, y)), self.hparams)

    def filter(self, predicate):
        return self.apply(tf.data.Dataset.filter(self.dataset, predicate), self.hparams)


class _FrontendZippedView(FrontendZippedViewBase):
    def __init__(self, zipped: tf.data.Dataset, hparams):
        self._dataset = zipped
        self._hparams = hparams

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def apply(self, dataset, hparams):
        return _FrontendZippedView(dataset, hparams)

    def group_by_batch(self):
        batch_size = self.hparams.batch_size
        approx_min_target_length = self.hparams.approx_min_target_length
        bucket_width = self.hparams.batch_bucket_width
        num_buckets = self.hparams.batch_num_buckets

        def key_func(source, target):
            target_length = tf.minimum(target.target_length - approx_min_target_length, 0)
            bucket_id = target_length // bucket_width
            return tf.minimum(tf.to_int64(num_buckets), bucket_id)

        def reduce_func(unused_key, window: tf.data.Dataset):
            # ToDo: use padded_batch instead of padded_batch_and_drop_remainder
            # Currently this model only works with static batch size
            apply_fn = tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes=(
                PreparedSourceData(
                    id=tf.TensorShape([]),
                    text=tf.TensorShape([]),
                    source=tf.TensorShape([None]),
                    source_length=tf.TensorShape([]),
                    text_positions=tf.TensorShape([None]),
                    text2=tf.TensorShape([]),
                    source2=tf.TensorShape([None]),
                    source_length2=tf.TensorShape([]),
                    text_positions2=tf.TensorShape([None]),
                ),
                _PreparedTargetData(
                    id=tf.TensorShape([]),
                    spec=tf.TensorShape([None, self.hparams.fft_size // 2 + 1]),
                    spec_width=tf.TensorShape([]),
                    mel=tf.TensorShape([None, self.hparams.num_mels]),
                    mel_width=tf.TensorShape([]),
                    target_length=tf.TensorShape([]),
                    done=tf.TensorShape([None]),
                )), padding_values=(
                PreparedSourceData(
                    id=tf.to_int64(0),
                    text="",
                    source=tf.to_int64(0),
                    source_length=tf.to_int64(0),
                    text_positions=tf.to_int64(0),
                    text2="",
                    source2=tf.to_int64(0),
                    source_length2=tf.to_int64(0),
                    text_positions2=tf.to_int64(0),
                ),
                _PreparedTargetData(
                    id=tf.to_int64(0),
                    spec=tf.to_float(0),
                    spec_width=tf.to_int64(0),
                    mel=tf.to_float(0),
                    mel_width=tf.to_int64(0),
                    target_length=tf.to_int64(0),
                    done=tf.to_float(1),
                )))
            return window.apply(apply_fn)

        batched = self.dataset.apply(tf.contrib.data.group_by_window(key_func,
                                                                     reduce_func,
                                                                     window_size=batch_size*5))
        return _FrontendBatchedView(batched, self.hparams)


class _FrontendBatchedViewBase(FrontendZippedViewBase):
    def add_memory_mask(self):
        def convert(s: PreparedSourceData, t):
            mask_value = -1e9

            def to_float_mask(mask):
                return tf.to_float(tf.logical_not(mask)) * mask_value

            s1_mask = to_float_mask(tf.sequence_mask(s.source_length, tf.shape(s.source)[1]))
            s2_mask = to_float_mask(tf.sequence_mask(s.source_length2, tf.shape(s.source2)[1]))

            return PreparedSourceDataWithMask(
                id=s.id,
                text=s.text,
                source=s.source,
                source_length=s.source_length,
                mask=s1_mask,
                text_positions=s.text_positions,
                text2=s.text2,
                source2=s.source2,
                source_length2=s.source_length2,
                text_positions2=s.text_positions2,
                mask2=s2_mask,
            ), t

        converted = self.dataset.map(lambda x, y: convert(x, y))
        return self.apply(converted, self.hparams)


class _FrontendBatchedView(_FrontendBatchedViewBase):
    def __init__(self, batched: tf.data.Dataset, hparams):
        self._dataset = batched
        self._hparams = hparams

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def apply(self, dataset, hparams):
        return _FrontendBatchedView(dataset, hparams)

    def add_frame_positions(self):
        r = self.hparams.outputs_per_step
        downsample_step = self.hparams.downsample_step

        def convert(source, target):
            max_decoder_target_len = tf.shape(target.mel)[1] // r // downsample_step
            frame_positions = tf.tile(tf.expand_dims(tf.range(1, max_decoder_target_len + 1), axis=0),
                                      [self.hparams.batch_size, 1])
            return source, PreparedTargetData(
                id=target.id,
                spec=target.spec,
                spec_width=target.spec_width,
                mel=target.mel,
                mel_width=target.mel_width,
                target_length=target.target_length,
                done=target.done,
                frame_positions=frame_positions,
            )

        converted = self.dataset.map(lambda x, y: convert(x, y))
        return _FrontendBatchedViewWithFramePositions(converted, self.hparams)


class _FrontendBatchedViewWithFramePositions(_FrontendBatchedViewBase):
    def __init__(self, batched: tf.data.Dataset, hparams):
        self._dataset = batched
        self._hparams = hparams

    @property
    def dataset(self):
        return self._dataset

    @property
    def hparams(self):
        return self._hparams

    def apply(self, dataset, hparams):
        return _FrontendBatchedViewWithFramePositions(dataset, hparams)

    def downsample_mel(self):
        def convert(source, target):
            return source, PreparedTargetDataWithMask(
                id=target.id,
                spec=target.spec,
                spec_width=target.spec_width,
                mel=target.mel[:, 0::self.hparams.downsample_step, :],
                mel_width=target.mel_width,
                target_length=target.target_length,
                done=target.done,
                frame_positions=target.frame_positions,
                spec_loss_mask=target.spec_loss_mask,
                binary_loss_mask=target.binary_loss_mask,
            )

        converted = self.dataset.map(lambda x, y: convert(x, y))
        return _FrontendBatchedViewWithFramePositions(converted, self.hparams)

    def add_target_mask(self):
        r = self.hparams.outputs_per_step
        downsample_step = self.hparams.downsample_step

        def convert(s, t: PreparedTargetData):

            def to_float_mask(mask):
                return tf.to_float(mask)

            spec_loss_mask = to_float_mask(tf.sequence_mask(t.target_length // downsample_step, tf.shape(t.mel)[1] // downsample_step))
            binary_loss_mask = to_float_mask(tf.sequence_mask(t.target_length // r // downsample_step, tf.shape(t.frame_positions)[1]))

            return s, PreparedTargetDataWithMask(
                id=t.id,
                spec=t.spec,
                spec_width=t.spec_width,
                mel=t.mel,
                mel_width=t.mel_width,
                target_length=t.target_length,
                done=t.done,
                frame_positions=t.frame_positions,
                spec_loss_mask=spec_loss_mask,
                binary_loss_mask=binary_loss_mask,
            )

        converted = self.dataset.map(lambda x, y: convert(x, y))
        return self.apply(converted, self.hparams)