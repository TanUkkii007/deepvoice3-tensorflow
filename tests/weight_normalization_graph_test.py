import tensorflow as tf
import numpy as np
import uuid
from hypothesis import given, assume, settings, unlimited
from hypothesis.strategies import integers
from hypothesis.extra.numpy import arrays
from deepvoice3.weight_normalization import weight_normalization, WeightNormalization


class WeightNormalizationTest(tf.test.TestCase):

    @given(_weight=arrays(dtype=np.float32, shape=[4, 5], elements=integers(-5, 5)),
           input=arrays(dtype=np.float32, shape=[5, 1], elements=integers(-10, 10), unique=True))
    @settings(max_examples=10, timeout=unlimited)
    def test_weight_normalization_2(self, _weight, input):
        assume(not np.any(np.all(_weight == 0.0, axis=1)))
        input_pf = tf.placeholder(dtype=tf.float32, shape=[5, 1])
        weight = tf.Variable(_weight, trainable=False)
        wn = WeightNormalization(weight, dimension=0)
        normalized_weight = wn.apply(weight)
        output = tf.reduce_sum(tf.matmul(normalized_weight, input_pf), axis=0)
        output_original = tf.reduce_sum(tf.matmul(weight, input_pf), axis=0)

        def zero_gradient_if_none(grad):
            return [tf.zeros_like(output) if g is None else g for g in grad]

        grad_g = zero_gradient_if_none(tf.gradients(output, wn.g))
        grad_v = zero_gradient_if_none(tf.gradients(output, wn.v))
        grad_w = zero_gradient_if_none(tf.gradients(output, weight))
        grad_w_original = zero_gradient_if_none(tf.gradients(output_original, weight))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            w, nw = sess.run([weight, normalized_weight])
            self.assertAllEqual(w, nw)

            output_value, grad_w_value, grad_g_value, grad_v_value, grad_w_original_value = sess.run(
                [output, grad_w, grad_g, grad_v, grad_w_original], feed_dict={input_pf: input})
            # assert no dependency on weight
            self.assertAllEqual(np.zeros_like(output_value), grad_w_value[0])

            v = _weight
            v_norm = np.linalg.norm(v, axis=1)
            # \nabla_g L = \sum_{i,j}\frac{\partial L}{\partial w_{ij}}\nabla_g w_{i,j}
            self.assertAllClose(grad_g_value[0],
                                np.sum(grad_w_original_value[0] * v, axis=1) / v_norm)

    @given(_weight=arrays(dtype=np.float32, shape=[3, 4, 5], elements=integers(-5, 5)),
           input=arrays(dtype=np.float32, shape=[3, 5, 1], elements=integers(-10, 10), unique=True))
    @settings(max_examples=10, timeout=unlimited)
    def test_weight_normalization_3(self, _weight, input):
        assume(not np.all(_weight == 0.0))
        input_pf = tf.placeholder(dtype=tf.float32, shape=[3, 5, 1])
        weight = tf.get_variable(str(uuid.uuid4()), shape=[3, 4, 5], initializer=tf.constant_initializer(_weight))
        wn = WeightNormalization(weight, dimension=0)
        normalized_weight = wn.apply(weight)
        output = tf.reduce_sum(tf.matmul(normalized_weight, input_pf), axis=0)
        output_original = tf.reduce_sum(tf.matmul(weight, input_pf), axis=0)

        def zero_gradient_if_none(grad):
            return [tf.zeros_like(output) if g is None else g for g in grad]

        grad_g = zero_gradient_if_none(tf.gradients(output, wn.g))
        grad_v = zero_gradient_if_none(tf.gradients(output, wn.v))
        grad_w = zero_gradient_if_none(tf.gradients(output, weight))
        grad_w_original = zero_gradient_if_none(tf.gradients(output_original, weight))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            output_value, grad_w_value, grad_g_value, grad_v_value, grad_w_original_value = sess.run(
                [output, grad_w, grad_g, grad_v, grad_w_original], feed_dict={input_pf: input})
            # assert no dependency on weight
            self.assertAllEqual(np.zeros_like(output_value), grad_w_value[0])

            v = _weight
            v_norm = np.linalg.norm(v, axis=(1, 2))
            # \nabla_g L = \sum_{i,j}\frac{\partial L}{\partial w_{ij}}\nabla_g w_{i,j}
            self.assertAllClose(grad_g_value[0],
                                np.sum(grad_w_original_value[0] * v, axis=(1,2)) / v_norm)