import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset, random_split
import glob

# load all CSV files in the folder and combine them
csv_files = glob.glob("training_data*.csv")
print(f"Found {len(csv_files)} data files: {csv_files}")
df = pd.concat([pd.read_csv(f, header=None) for f in csv_files], ignore_index=True)
print(df.iloc[:, 0].unique()[:20])  # show first 20 unique values in label column
print(df.head(5))

df = df.dropna()                          # remove nan rows
df.iloc[:, 0] = df.iloc[:, 0].astype(float).astype(int)  # convert 0.0 -> 0
df = df[df.iloc[:, 0].between(0, 6)]     # keep only valid class labels 0-6

print(f"After cleaning: {len(df)} samples")

X = df.iloc[:, 1:].values.astype(np.float32)
y = df.iloc[:, 0].values.astype(np.int64)

print(f"Loaded {len(X)} samples, {X.shape[1]} features, {len(set(y))} classes")

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)
dataset = TensorDataset(X_tensor, y_tensor)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32)

class MeasuresNetworkKodalyC1C2_slim(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(275, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1024)
        self.fc5 = nn.Linear(1024, 7)
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            init.xavier_uniform_(fc.weight)
            init.constant_(fc.bias, 0.0)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.dropout(torch.relu(self.fc2(x)))
        x = self.dropout(torch.relu(self.fc3(x)))
        x = self.dropout(torch.relu(self.fc4(x)))
        return self.fc5(x)

model = MeasuresNetworkKodalyC1C2_slim()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(20):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()

    model.eval()
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = torch.argmax(model(X_batch), dim=1)
            correct += (preds == y_batch).sum().item()
    acc = correct / val_size
    print(f"Epoch {epoch+1}/20 — val accuracy: {acc:.2%}")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
checkpoint_name = f"checkpoints/checkpoint_{timestamp}.pt"
torch.save({'model_state_dict': model.state_dict()}, checkpoint_name)
print(f"Saved new checkpoint: {checkpoint_name}")

torch.save({'model_state_dict': model.state_dict()}, "checkpoints/checkpoint_latest.pt")
print("Also saved as checkpoint_latest.pt")