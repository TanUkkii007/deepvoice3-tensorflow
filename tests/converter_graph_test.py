import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited
from hypothesis.strategies import integers, floats, composite
from hypothesis.extra.numpy import arrays
from deepvoice3.deepvoice3 import Converter

@composite
def input_tensor(draw, b_size=integers(2, 5), input_length_size=integers(1, 20), in_features_size=integers(1, 20), out_features_size=integers(1, 10), elements=floats(0.0, 1.0)):
    batch_size = draw(b_size)
    input_length = draw(input_length_size)
    in_features = draw(in_features_size)
    out_features = draw(out_features_size)
    input = draw(arrays(dtype=np.float32, shape=[batch_size, input_length, in_features], elements=elements))
    return (batch_size, input_length, in_features, out_features, input)

class ConverterTest(tf.test.TestCase):

    @given(inputs=input_tensor(), seed=integers(0, 1234567))
    @settings(max_examples=10, timeout=unlimited)
    def test_converter(self, inputs, seed):
        tf.set_random_seed(seed)
        batch_size, input_length, in_features, out_features, input = inputs
        input = tf.convert_to_tensor(input)
        converter = Converter(in_features, out_features, dropout=0.0)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            converter(input)