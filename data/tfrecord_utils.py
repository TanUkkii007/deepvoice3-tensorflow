import tensorflow as tf
import numpy as np
from data import PreprocessedData


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_tfrecord(example: tf.train.Example, filename: str):
    with tf.python_io.TFRecordWriter(filename) as writer:
        writer.write(example.SerializeToString())


def write_preprocessed_data(text: str, spec: np.ndarray, mel: np.ndarray, filename: str):
    raw_spec = spec.tostring()
    raw_mel = mel.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'text': bytes_feature(text.encode('utf-8')),
        'spec': bytes_feature(raw_spec),
        'spec_width': int64_feature(spec.shape[1]),
        'mel': bytes_feature(raw_mel),
        'mel_width': int64_feature(mel.shape[1]),
        'target_length': int64_feature(len(mel)),
    }))
    write_tfrecord(example, filename)


def read_preprocessed_data(filename):
    record_iterator = tf.python_io.tf_record_iterator(filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        text = example.features.feature['text'].bytes_list.value[0].decode("utf-8")
        spec = example.features.feature['spec'].bytes_list.value[0]
        mel = example.features.feature['mel'].bytes_list.value[0]
        spec_width = example.features.feature['spec_width'].int64_list.value[0]
        mel_width = example.features.feature['mel_width'].int64_list.value[0]
        target_length = example.features.feature['target_length'].int64_list.value[0]
        spec = np.frombuffer(spec, dtype=np.float32).reshape([target_length, spec_width])
        mel = np.frombuffer(mel, dtype=np.float32).reshape([target_length, mel_width])
        yield PreprocessedData(
            text=text,
            spec=spec,
            spec_width=spec_width,
            mel=mel,
            mel_width=mel_width,
            target_length=target_length,
        )


def parse_preprocessed_data(proto):
    features = {
        'text': tf.FixedLenFeature((), tf.string),
        'spec': tf.FixedLenFeature((), tf.string),
        'spec_width': tf.FixedLenFeature((), tf.int64),
        'mel': tf.FixedLenFeature((), tf.string),
        'mel_width': tf.FixedLenFeature((), tf.int64),
        'target_length': tf.FixedLenFeature((), tf.int64),
    }
    parsed_features = tf.parse_single_example(proto, features)
    return parsed_features


def decode_preprocessed_data(parsed):
    spec_width = parsed['spec_width']
    mel_width = parsed['mel_width']
    target_length = parsed['target_length']
    spec = tf.decode_raw(parsed['spec'], tf.float32)
    mel = tf.decode_raw(parsed['mel'], tf.float32)
    return PreprocessedData(
        text=parsed['text'],
        spec=tf.reshape(spec, shape=tf.stack([target_length, spec_width], axis=0)),
        spec_width=spec_width,
        mel=tf.reshape(mel, shape=tf.stack([target_length, mel_width], axis=0)),
        mel_width=mel_width,
        target_length=target_length,
    )
