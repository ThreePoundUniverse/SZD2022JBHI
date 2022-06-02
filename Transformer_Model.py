

"""
The code is modified on the basis of vit-keras [https://github.com/faustomorales/vit-keras]
The code is a simplified version of the original code. For any inconvernienes and bug reports, contact syuri@tju.edu.cn
"""


import tensorflow as tf
from tensorflow.keras import layers



class GAP1Dtime(layers.Layer):
    def build(self, input_shape):
        assert (
            len(input_shape) == 4
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
    def call(self, inputs):
        return tf.reduce_mean(inputs, 1)


class ClassToken(layers.Layer):
    """Append a class token to an input layer."""
    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="cls",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)


class AddPositionEmbs(layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""
    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = layers.Dense(hidden_size, name="query")
        self.key_dense = layers.Dense(hidden_size, name="key")
        self.value_dense = layers.Dense(hidden_size, name="value")
        self.combine_heads = layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                ),
                layers.Dropout(self.dropout),
                layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout = layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights


def build_model(channels: int, samplepoints: int, classes: int, activation='sigmoid',
                num_layers=2, num_heads=4, hidden_size=128, mlp_dim=256, dropout=0.1):
    ## input ##
    x = layers.Input(shape=(channels, samplepoints), name='input')
    y = layers.Permute((2, 1), input_shape=(x.shape[1], x.shape[2]), name='permute_1')(x)
    y = layers.Reshape((y.shape[1], y.shape[2], 1), name='reshape_1')(y)
    ## conv 1 ##
    y = layers.Conv2D(32, (4, 1), strides=(2, 1), activation='relu', padding='same', name='cnn_1')(y)
    y = layers.BatchNormalization()(y)
    ## conv 2 ##
    y = layers.Conv2D(64, (4, 1), strides=(2, 1), activation='relu', padding='same', name='cnn_2')(y)
    y = layers.BatchNormalization()(y)
    ## conv 3 ##
    y = layers.Conv2D(hidden_size, (4, 1), strides=(2, 1), activation='relu', padding='same', name='cnn_3')(y)
    ## avg pooling ##
    y = GAP1Dtime(name='gap1Dtime')(y)
    ## add extra learnable class embedding ##
    y = ClassToken(name='class_token')(y)
    ## add position embedding ##
    y = AddPositionEmbs(name='posembed_input')(y)
    ## transformer block ##
    for n in range(num_layers):
        y, _ = TransformerBlock(num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, name=f'transformer_{n}')(y)
    y = layers.LayerNormalization(epsilon=1e-6, name="transformer_norm")(y)
    y = layers.Lambda(lambda v: v[:, 0], name='extract_token')(y)
    ## classifier ##
    y = layers.Dense(classes, name='head', activation=activation)(y)
    return tf.keras.models.Model(inputs=x, outputs=y)


if __name__ =='__main__':
    model = build_model(channels=24, samplepoints=512, classes=2, activation='sigmoid')
    model.summary()