import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import joblib
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data_folder = 'data/stocks'
model_file = os.path.join('models', 'best_lstm_stock.pth')

X_test = np.load(os.path.join(data_folder, 'X_test.npy'))
y_test = np.load(os.path.join(data_folder, 'y_test.npy'))

scaler_y = joblib.load(os.path.join(data_folder, 'scaler_y.gz'))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size) 

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

num_features = X_test.shape[2]
model = LSTMModel(input_size=num_features, hidden_size=256, num_layers=3, output_size=1).to(device)
model.load_state_dict(torch.load(model_file, map_location=device))
model.eval()


X_test_tensor = torch.from_numpy(X_test).float().to(device)

with torch.no_grad():
    predictions_scaled = model(X_test_tensor).cpu().numpy()

predictions_unscaled = scaler_y.inverse_transform(predictions_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(np.mean((predictions_unscaled - y_test_unscaled)**2))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

print("Beberapa harga asli (USD) dari data pengujian:")
for i in range(10):
    print(f"Hari {i+1}: Actual ${y_test_unscaled[i,0]:.2f} | Predicted ${predictions_unscaled[i,0]:.2f}")

print("------------------------")

plt.figure(figsize=(16, 8))
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Adj Close Price USD ($)')
plt.plot(y_test_unscaled, color='blue', label='Actual Price')
plt.plot(predictions_unscaled, color='red', label='Predicted Price')
plt.legend()
plt.show()
