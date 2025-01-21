import argparse
import numpy as np
from sklearn.preprocessing import LabelEncoder
from data_preprocessing import load_data, split_transform
from train_evaluate import train_cnn_lstm, train_dnn, train_gru, train_svm


def main():
    parser = argparse.ArgumentParser(description="Run models with different architectures.")
    parser.add_argument("--model", type=str, required=True, choices=["cnn_lstm_transformer", "dnn", "gru", "svm"],
                        help="Choose the model to train and evaluate.")
    args = parser.parse_args()

    # Load data
    df = load_data()
    X_sc_train, X_sc_test, y_train, y_test, scaler = split_transform(df, split_ratio=0.3)
    encoder = LabelEncoder()
    encoder.fit(y_train)

    # Model selection and training
    if args.model == "cnn_lstm_transformer":
        input_shape = (X_sc_train.shape[1], 1)
        output_shape = len(encoder.classes_)
        X_sc_train = np.expand_dims(X_sc_train, axis=2)
        X_sc_test = np.expand_dims(X_sc_test, axis=2)
        train_cnn_lstm(X_sc_train, X_sc_test, y_train, y_test, encoder, input_shape, output_shape)

    elif args.model == "dnn":
        input_shape = (X_sc_train.shape[1],)
        output_shape = len(encoder.classes_)
        train_dnn(X_sc_train, X_sc_test, y_train, y_test, encoder, input_shape, output_shape)

    elif args.model == "gru":
        input_shape = (X_sc_train.shape[1], 1)
        output_shape = len(encoder.classes_)
        X_sc_train = np.expand_dims(X_sc_train, axis=2)
        X_sc_test = np.expand_dims(X_sc_test, axis=2)
        train_gru(X_sc_train, X_sc_test, y_train, y_test, encoder, input_shape, output_shape)

    elif args.model == "svm":
        train_svm(X_sc_train, X_sc_test, y_train, y_test, encoder)


if __name__ == "__main__":
    main()
