import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from data_processing import get_data
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
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
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# def build_model(input_shape, num_classes):
#     inputs = layers.Input(shape=input_shape)
#     x = inputs
#     transformer_block = TransformerBlock(embed_dim=32, num_heads=2, ff_dim=32)
#     x = transformer_block(x)
#     x = layers.GlobalAveragePooling1D()(x)
#     x = layers.Dropout(0.1)(x)
#     outputs = layers.Dense(num_classes, activation="softmax")(x)
#     return tf.keras.Model(inputs=inputs, outputs=outputs)

def build_model(input_shape, embed_dim, num_heads, ff_dim, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)



def load_and_preprocess_data():
    features, labels = get_data()
    return features, labels

def compile_and_train_model(model, features, labels):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    # Split the features and labels into training and validation sets
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Train the model
    history = model.fit(
        train_features,
        train_labels,
        batch_size=32,
        epochs=10,
        validation_data=(val_features, val_labels),
    )
    return history

def main():
    # Load and preprocess data
    features, labels = load_and_preprocess_data()

    # Build model
    # model = build_model(input_shape=(features.shape[1],), num_classes=3)
    model = build_model(input_shape=(features.shape[1],), embed_dim=features.shape[1], num_heads=2, ff_dim=32,
                        num_classes=3)

    # Compile and train model
    history = compile_and_train_model(model, features, labels)

    # Evaluate model
    test_loss, test_acc = model.evaluate(features, labels, verbose=2)
    print(f"\nTest accuracy: {test_acc}")

if __name__ == "__main__":
    main()
