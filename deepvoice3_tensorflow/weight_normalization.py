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
        g_shape = [weight_shape[i].value for i in self.reduction_axis]
        self.g = self.add_variable(name="g", shape=g_shape, dtype=tf.float32,
                                   initializer=lambda shape, dtype=None, partition_info=None,
                                                      verify_shape=None: self.initial_g())
        self.v = self.add_variable(name="v", shape=weight_shape, dtype=tf.float32,
                                   initializer=lambda shape, dtype=None, partition_info=None,
                                                      verify_shape=None: self.weight_value)
        self.built = True

    def call(self, _, training=False):
        return self.compute_weight()

    def initial_g(self):
        norm = tf.norm(self.weight_value, axis=self._unwrap_if_rank0(self.reduction_axis))
        return norm

    def compute_weight(self):
        g = self.g
        v = self.v
        gv = (g / tf.norm(v, axis=self._unwrap_if_rank0(self.reduction_axis)))
        # expand dimension in addition to batch. ToDo: generalize
        gv = tf.reshape(gv, shape=[-1] + [1 for _ in range(self.ndims - 1)])
        return v * gv

    def _compute_reduction_axis(self):
        return [i for i in range(0, self.ndims) if i != self.dimension]

    def _unwrap_if_rank0(self, r):
        return r[0] if len(r) == 1 else r

def weight_normalization(weight, dimension=0):
    wn = WeightNormalization(weight, dimension)
    return wn.apply(weight)
