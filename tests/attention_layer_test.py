import tensorflow as tf
import numpy as np
from hypothesis import given, settings, unlimited, assume
from hypothesis.strategies import integers, floats, composite
from hypothesis.extra.numpy import arrays
from deepvoice3_tensorflow.deepvoice3 import ScaledDotProductAttentionMechanism, AttentionLayer


@composite
def attention_tensors(draw, b_size=integers(1, 5), t_query_size=integers(2, 20), c_size=integers(1, 10),
                      t_encoder_size=integers(2, 10), embed_dim=integers(1, 10), elements=integers(-5, 5)):
    b = draw(b_size)
    t_query = draw(t_query_size)
    c = draw(c_size)
    t_encoder = draw(t_encoder_size)
    embed = draw(embed_dim)
    query = draw(arrays(dtype=np.float32, shape=[b, t_query, c], elements=elements))
    encoder = draw(arrays(dtype=np.float32, shape=[b, t_encoder, embed], elements=elements))
    return (b, t_query, c, t_encoder, embed, query, encoder)


class AttentionLayerTest(tf.test.TestCase):

    @given(tensors=attention_tensors(), dropout=floats(0.5, 1.0, allow_nan=False))
    @settings(max_examples=10, timeout=unlimited)
    def test_attention(self, tensors, dropout):
        B, T_query, C, T_encoder, embed_dim, query, encoder_out = tensors
        assume(B * T_query * C > 1)
        assume(B * T_encoder * embed_dim > 1)
        query = tf.constant(query)
        encoder_out = tf.constant(encoder_out)

        attention_mechanism = ScaledDotProductAttentionMechanism(encoder_out, embed_dim)

        attention = AttentionLayer(attention_mechanism, C, dropout)
        output = attention.apply(query)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(output)
