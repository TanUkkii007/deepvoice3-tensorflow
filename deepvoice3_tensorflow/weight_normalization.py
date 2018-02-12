import tensorflow as tf


class WeightNormalization(tf.layers.Layer):
    def __init__(self, weight, dimension, trainable=True, name=None, **kwargs):
        super(WeightNormalization, self).__init__(
            trainable=trainable, name=name, **kwargs)
        self.weight_value = weight.initialized_value()
        self.dimension = dimension
        self.ndims = self.weight_value.shape.ndims
        self.reduction_axis = self._compute_reduction_axis()

    def build(self, weight_shape):
        # add g and v as new parameters and express w as g/||v|| * v
        self.g = tf.Variable(self.initial_g(), name="g")
        self.v = tf.Variable(self.weight_value, name="v")
        self.built = True

    def call(self, _, training=False):
        return self.compute_weight()

    def initial_g(self):
        norm = tf.norm(self.weight_value, axis=self.reduction_axis)
        return norm

    def compute_weight(self):
        g = self.g
        v = self.v
        gv = (g / tf.norm(v, axis=self.reduction_axis))
        # expand dimension in addition to batch. ToDo: generalize
        gv = tf.reshape(gv, shape=[-1] + [1 for _ in range(self.ndims - 1)])
        return v * gv

    def _compute_reduction_axis(self):
        r = [i for i in range(0, self.ndims) if i != self.dimension]
        return r[0] if len(r) == 1 else r

def weight_normalization(weight, dimension=0):
    wn = WeightNormalization(weight, dimension)
    return wn.apply(weight)
