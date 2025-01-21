# CNN-LSTMFORMER

This project provides a  implementation of different machine learning models for training and evaluation, including:
- CNN-LSTM-Transformer
- DNN (Deep Neural Network)
- GRU (Gated Recurrent Unit)
- SVM (Support Vector Machine)

## Project Structure

- **data_preprocessing.py**: Contains functions for loading and preprocessing the data.
- **dnn_model.py**: Defines the DNN model architecture.
- **gru_model.py**: Defines the GRU model architecture.
- **model_definition.py**: Contains the CNN-LSTM-Transformer model and its components.
- **train_evaluate.py**: Contains functions for training and evaluating each model.
- **main.py**: Entry point for selecting and running a model.
- **results/**: Folder where the results (e.g., accuracy, classification reports, confusion matrix plots) will be saved.

## Usage

### Install Dependencies
Ensure that the required Python packages are installed. Use the following command:
```bash
pip install tensorflow scikit-learn matplotlib pandas numpy
```

### Prepare the Data
Place the required data files (`Max_solar_data.csv` and `Lim_solar_data.csv`) in the root directory.

---

## Run a Model
Use the following command to train and evaluate a specific model:
```bash
python main.py --model <model_name>
``` 
where `<model_name>` can be one of the following:
- `dnn`
- `gru`
- `cnn_lstm_transformer`
- `svm`

For example, to run the CNN-LSTM-Transformer model, use the following command:
```bash
python main.py --model cnn_lstm_transformer
```

---