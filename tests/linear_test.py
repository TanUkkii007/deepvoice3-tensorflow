import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited
from hypothesis.strategies import integers, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.modules import Linear

@composite
def input_tensor(draw, b_size=integers(2, 5), input_length_size=integers(1, 20), in_features_size=integers(1, 20), out_features_size=integers(1, 10), elements=integers(-5, 5)):
    batch_size = draw(b_size)
    input_length = draw(input_length_size)
    in_features = draw(in_features_size)
    out_features = draw(out_features_size)
    input = draw(arrays(dtype=np.float32, shape=[batch_size, input_length, in_features], elements=elements))
    return (batch_size, input_length, in_features, out_features, input)

class LinearTest(tf.test.TestCase):

    @given(inputs=input_tensor(), seed=integers(0, 1234567))
    @settings(max_examples=10, timeout=unlimited)
    def test_linear(self, inputs, seed):
        batch_size, input_length, in_features, out_features, input = inputs
        linear = Linear(in_features, out_features, dropout=1.0, weight_initializer_seed=seed)
        batches = [linear(tf.constant(b)) for b in [input[i] for i in range(0, batch_size)]]
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            x, y = sess.run([tf.stack(batches), linear(tf.constant(input))])
            self.assertAllClose(x, y)