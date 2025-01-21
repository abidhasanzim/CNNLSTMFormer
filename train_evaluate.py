import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from model_definition import CNNLSTformer_model
from dnn_model import DNN_model
from gru_model import GRU_model


def save_results(results_dir, accuracy, classification_rep, cm, prefix):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "results.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        f.write("\nClassification Report:\n")
        f.write(classification_rep)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f"{results_dir}/{prefix}_confusion_matrix.png")
    plt.close()


def train_cnn_lstm(X_sc_train, X_sc_test, y_train, y_test, encoder, input_shape, output_shape):
    results_dir = "results/cnn_lstm_transformer"
    dummy_y = to_categorical(encoder.transform(y_train))
    dummy_y_test = to_categorical(encoder.transform(y_test))
    model = CNNLSTformer_model(input_shape, output_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_sc_train, dummy_y, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_sc_test, dummy_y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_sc_test), axis=-1)
    y_true = np.argmax(dummy_y_test, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, target_names=encoder.classes_)
    save_results(results_dir, test_accuracy, classification_rep, cm, "cnn_lstm")
    return history


def train_dnn(X_train, X_test, y_train, y_test, encoder, input_shape, output_shape):
    results_dir = "results/dnn"
    dummy_y = to_categorical(encoder.transform(y_train))
    dummy_y_test = to_categorical(encoder.transform(y_test))
    model = DNN_model(input_shape, output_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_train, dummy_y, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_test, dummy_y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    y_true = np.argmax(dummy_y_test, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, target_names=encoder.classes_)
    save_results(results_dir, test_accuracy, classification_rep, cm, "dnn")
    return history


def train_gru(X_sc_train, X_sc_test, y_train, y_test, encoder, input_shape, output_shape):
    results_dir = "results/gru"
    dummy_y = to_categorical(encoder.transform(y_train))
    dummy_y_test = to_categorical(encoder.transform(y_test))
    model = GRU_model(input_shape, output_shape)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(X_sc_train, dummy_y, epochs=10, batch_size=64, validation_split=0.2, verbose=1)
    test_loss, test_accuracy = model.evaluate(X_sc_test, dummy_y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_sc_test), axis=-1)
    y_true = np.argmax(dummy_y_test, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, target_names=encoder.classes_)
    save_results(results_dir, test_accuracy, classification_rep, cm, "gru")
    return history


def train_svm(X_train, X_test, y_train, y_test, encoder):
    results_dir = "results/svm"
    svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", decision_function_shape="ovo")
    svm_model.fit(X_train, y_train)
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=encoder.classes_)
    save_results(results_dir, accuracy, classification_rep, cm, "svm")
