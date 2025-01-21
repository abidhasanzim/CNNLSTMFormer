
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def split_transform(df, split_ratio):
    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=split_ratio, shuffle=True)

    scaler = StandardScaler()
    X_sc_train = scaler.fit_transform(X_train)
    X_sc_test = scaler.transform(X_test)

    return X_sc_train, X_sc_test, y_train, y_test, scaler

def load_data():
    df1 = pd.read_csv('Max_solar_data.csv')
    df2 = pd.read_csv('Lim_solar_data.csv')
    df = pd.concat([df1, df2], ignore_index=True, axis=0)
    return df
