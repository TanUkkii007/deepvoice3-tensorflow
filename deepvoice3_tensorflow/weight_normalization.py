import tensorflow as tf


class WeightNormalization(tf.layers.Layer):
    def __init__(self, weight_value, dimension, trainable=True, name=None, **kwargs):
        super(WeightNormalization, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self.weight_value = weight_value
        self.dimension = dimension

    def build(self, weight_shape):
        # add g and v as new parameters and express w as g/||v|| * v
        self.g = tf.Variable(self.g_norm(), name="g")
        self.v = tf.Variable(self.weight_value, name="v")
        self.built = True

    def call(self, weight, training=False):
        return self.compute_weight()

    def g_norm(self):
        norm = tf.norm(self.weight_value, axis=self.dimension)
        return norm

    def compute_weight(self):
        g = self.g
        v = self.v
        gv = (g / tf.norm(v, axis=self.dimension))
        # expand dimension in addition to batch
        gv = tf.reshape(gv, shape=[-1] + [1 for _ in self.dimension])
        return v * gv


def weight_normalization(weight, dimension=(1,2)):
    wn = WeightNormalization(weight.initialized_value(), dimension)
    return wn.apply(weight)
