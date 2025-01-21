from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def DNN_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    x = Dense(32, activation="relu")(inputs)
    x = Dense(64, activation="relu")(x)
    x = Dense(128, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(output_shape, activation="softmax")(x)
    model = Model(inputs, outputs)
    return model
