import tensorflow as tf
import numpy as np
from hypothesis import given, assume, settings, unlimited
from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.weight_normalization import weight_normalization, WeightNormalization


class WeightNormalizationTest(tf.test.TestCase):

    @given(weight=arrays(dtype=np.float32, shape=[3,4,5], elements=integers(-5, 5)),
           input=arrays(dtype=np.float32, shape=[3, 5, 1], elements=integers(-10, 10), unique=True))
    @settings(max_examples=10, timeout=unlimited)
    def test_weight_normalization(self, weight, input):
        assume(not np.all(weight == 0.0))
        input_pf = tf.placeholder(dtype=tf.float32, shape=[3,5,1])
        weight = tf.Variable(weight, trainable=False)
        wn = WeightNormalization(weight.initialized_value(), dimension=(1,2))
        normalized = wn.apply(weight)
        output = tf.matmul(weight, input_pf)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            print(sess.run(normalized))