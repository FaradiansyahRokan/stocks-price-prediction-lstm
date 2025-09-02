import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

ticker_symbol = 'AAPL'
data_folder = 'data/stocks'
file_path = os.path.join(data_folder, f"{ticker_symbol.lower()}.csv")

try:
    df = pd.read_csv(file_path)

    features = ['Adj Close', 'Volume', 'Open', 'High', 'Low']
    data = df[features].values

 
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))


    X_features = data[:, 1:]
    X_scaled = scaler_X.fit_transform(X_features)

    y_target = data[:, 0].reshape(-1, 1)
    y_scaled = scaler_y.fit_transform(y_target)


    scaled_data = np.concatenate([y_scaled, X_scaled], axis=1)

    sequence_length = 60
    X_data, y_data = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_data.append(scaled_data[i-sequence_length:i, :])
        y_data.append(scaled_data[i, 0])

    X_data, y_data = np.array(X_data), np.array(y_data)

    training_data_len = int(np.ceil(len(X_data) * 0.8))
    X_train = X_data[:training_data_len]
    y_train = y_data[:training_data_len]
    X_test = X_data[training_data_len:]
    y_test = y_data[training_data_len:]

    np.save(os.path.join(data_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(data_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(data_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(data_folder, 'y_test.npy'), y_test)

    joblib.dump(scaler_X, os.path.join(data_folder, 'scaler_X.gz'))
    joblib.dump(scaler_y, os.path.join(data_folder, 'scaler_y.gz'))

    print(f"✅ Data untuk ticker {ticker_symbol} berhasil disiapkan dan disimpan.")

except FileNotFoundError:
    print(f"Error: File '{file_path}' tidak ditemukan.")
except KeyError as e:
    print(f"Error: Kolom {e} tidak ditemukan. Pastikan nama kolom di file CSV sudah benar.")
