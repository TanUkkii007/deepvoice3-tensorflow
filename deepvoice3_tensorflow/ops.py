import tensorflow as tf


def time_to_batch(value, dilation, name='time_to_batch'):
    '''

    :param value: (T, B, C)
    :param dilation:
    :param name:
    :return: (T * dilation, B / dilation, C)
    '''
    with tf.name_scope(name):
        shape = tf.shape(value)
        shape_t, shape_b, shape_c = shape[0], shape[1], shape[2]
        pad_elements = dilation - 1 - (shape_b + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape_c])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape_t * dilation, -1, shape_c])


def batch_to_time(value, dilation, name='batch_to_time'):
    '''

    :param value: (B, T, C)
    :param dilation:
    :param name:
    :return: (B / dilation, T * dilation, C)
    '''
    with tf.name_scope(name):
        shape = tf.shape(value)
        shape_b, shape_t, shape_c = shape[0], shape[1], shape[2]
        prepared = tf.reshape(value, [dilation, -1, shape_c])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])  # (B * T / dilation, dilation, C)
        return tf.reshape(transposed, [tf.div(shape_b, dilation), -1, shape_c])


def causal_conv(value, filter_, dilation, name='causal_conv'):
    '''

    :param value: (T, B, C)
    :param filter_:
    :param dilation:
    :param name:
    :return:
    '''
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
            restored = batch_to_time(conv, dilation)
        else:
            # dilation=1 has no effect to time_to_batch/batch_to_time transform
            restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')
        # Remove excess elements at the end.
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
        return result

