# CNN-LSTMFORMER

This project provides a  implementation of different machine learning models for training and evaluation, including:
- CNN-LSTM-Transformer
- DNN (Deep Neural Network)
- GRU (Gated Recurrent Unit)
- SVM (Support Vector Machine)



### GPVS-Faults Dataset

The **GPVS-Faults** dataset is a crucial resource for advancing fault detection and diagnosis in grid-connected photovoltaic (PV) systems. It simulates real-world fault scenarios under MPPT and IPPT modes, capturing varying fault severities, environmental disturbances, and noisy measurements. This dataset enables researchers to develop and validate algorithms for early fault detection, preventing system failures and enhancing the reliability of PV systems, contributing to the broader adoption of sustainable energy solutions.



### Data Preprocessing Overview

The solar panel dataset was preprocessed by combining data from multiple CSV files (`F0L` to `F7L` for limited power, and `F0M` to `F7M` for maximum power) into two consolidated DataFrames. A `label` column was added to identify the source file for each row. The data was sampled to reduce size, retaining every 1000th row for limited power data and every 100th row for maximum power data. Scatter plots were generated for each feature against time, with points styled by label for better visualization. Finally, the processed datasets were exported as `Lim_solar_data.csv` and `Max_solar_data.csv` for further analysis.


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

### Model Performance Comparison

The table below presents the average performance metrics for different models evaluated on the GPVS-Faults dataset. The **CNNLSTMFormer** model, developed as part of this project, achieves the best overall performance, highlighting its effectiveness in detecting and diagnosing faults in photovoltaic systems.

| Model           | Precision | Recall | F1-Score | Accuracy |
|------------------|-----------|--------|----------|----------|
| **CNNLSTMFormer** | **0.97**  | **0.97** | **0.97**  | **0.97**  |
| SVM             | 0.93      | 0.93   | 0.93     | 0.94     |
| DNN             | 0.95      | 0.96   | 0.96     | 0.95     |
| GRU             | 0.95      | 0.95   | 0.95     | 0.95     |

The **CNNLSTMFormer** combines the strengths of Convolutional Neural Networks (CNNs), Long Short-Term Memory networks (LSTMs), and Transformer architectures, enabling robust and accurate fault detection under varying conditions. 

This comparison validates the superiority of our proposed model in terms of precision, recall, F1-score, and accuracy. For detailed implementation and usage, refer to the project documentation.


---