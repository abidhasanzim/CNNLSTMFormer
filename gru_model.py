from tensorflow.keras.layers import GRU, Dense, Input
from tensorflow.keras.models import Model

def GRU_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = GRU(32, activation="tanh", return_sequences=True)(inputs)
    x = GRU(64, activation="tanh", return_sequences=True)(x)
    x = GRU(16, activation="tanh", return_sequences=False)(x)
    x = Dense(8, activation="selu")(x)
    outputs = Dense(output_shape, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
