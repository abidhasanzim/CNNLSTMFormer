
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, Dropout, LayerNormalization
from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def CNNLSTformer_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(filters=8, kernel_size=1, activation="relu")(inputs)
    x = Conv1D(filters=16, kernel_size=1, activation="relu")(x)
    x = Conv1D(filters=32, kernel_size=1, activation="relu")(x)
    x = LSTM(32, activation="tanh", return_sequences=True)(x)
    x = Conv1D(filters=64, kernel_size=1, activation="relu")(x)
    x = transformer_encoder(x, head_size=256, num_heads=2, ff_dim=4, dropout=0.025)
    x = Conv1D(filters=64, kernel_size=1, activation="relu")(x)
    x = Conv1D(filters=32, kernel_size=1, activation="relu")(x)
    x = LSTM(16, activation="tanh", return_sequences=False)(x)
    x = Dense(8, activation="selu")(x)
    outputs = Dense(output_shape, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
