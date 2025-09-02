import numpy as np
import torch
import joblib
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Menggunakan perangkat: {device}")

# === Load data ===
data_folder = 'data/stocks'
X_train = np.load(os.path.join(data_folder, 'X_train.npy'))
y_train = np.load(os.path.join(data_folder, 'y_train.npy'))


from sklearn.preprocessing import MinMaxScaler
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
y_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

X_tensor = torch.from_numpy(X_scaled).float().to(device)
y_tensor = torch.from_numpy(y_scaled).float().to(device)

joblib.dump(scaler_X, os.path.join(data_folder, "scaler_X.gz"))
joblib.dump(scaler_y, os.path.join(data_folder, "scaler_y.gz"))

# === Dataset split ===
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) 
        return out

num_features = X_tensor.shape[2]
model = LSTMModel(input_size=num_features, hidden_size=256, num_layers=3, output_size=1).to(device)

criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)


epochs = 200
best_val_loss = float("inf")
patience = 20
early_stop_counter = 0

for epoch in range(epochs):
    model.train()
    train_losses = []
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    model.eval()
    val_losses = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())

    avg_train_loss = np.mean(train_losses)
    avg_val_loss = np.mean(val_losses)

    scheduler.step(avg_val_loss)

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), os.path.join("models", "best_lstm_stock.pth"))
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("⏹ Early stopping triggered!")
            break

print("Training selesai. Model terbaik disimpan di 'models/best_lstm_stock.pth'")
