import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import glob
from torch.utils.data import TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── Model ─────────────────────────────────────────────────────────────────────
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
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        return self.fc5(x)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT     = "checkpoints/checkpoint_latest.pt"
CSV_PATTERN    = "training_data*.csv"
RANDOM_SEED    = 42          # fixed seed — reproducible split
TRAIN_FRACTION = 0.8         # must match what you used during training
CLASS_NAMES    = ["Do (C)", "Re (D)", "Mi (E)", "Fa (F)", "Sol (G)", "La (A)", "Si (B)"]

# ── Load & clean data ─────────────────────────────────────────────────────────
csv_files = glob.glob(CSV_PATTERN)
print(f"Found {len(csv_files)} CSV file(s): {csv_files}")

df = pd.concat([pd.read_csv(f, header=None) for f in csv_files], ignore_index=True)
df = df.dropna()
df.iloc[:, 0] = df.iloc[:, 0].astype(float).astype(int)
df = df[df.iloc[:, 0].between(0, 6)]
print(f"Total samples after cleaning: {len(df)}")

X = torch.tensor(df.iloc[:, 1:].values.astype(np.float32))
y = torch.tensor(df.iloc[:, 0].values.astype(np.int64))

# ── Reproduce the same 80/20 split used during training ──────────────────────
dataset    = TensorDataset(X, y)
train_size = int(TRAIN_FRACTION * len(dataset))
val_size   = len(dataset) - train_size

generator  = torch.Generator().manual_seed(RANDOM_SEED)
_, val_set = random_split(dataset, [train_size, val_size], generator=generator)

X_val  = torch.stack([val_set[i][0] for i in range(len(val_set))])
y_true = np.array([val_set[i][1].item() for i in range(len(val_set))])
print(f"Evaluating on {len(y_true)} validation samples ({int((1-TRAIN_FRACTION)*100)}% split)\n")

# ── Load model ────────────────────────────────────────────────────────────────
model = MeasuresNetworkKodalyC1C2_slim()
checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint: {CHECKPOINT}\n")

# ── Inference ─────────────────────────────────────────────────────────────────
with torch.no_grad():
    y_pred = torch.argmax(model(X_val), dim=1).numpy()

# ── Metrics ───────────────────────────────────────────────────────────────────
acc = accuracy_score(y_true, y_pred)
print(f"Overall Accuracy: {acc:.2%}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax
)
ax.set_xlabel("Predicted Note")
ax.set_ylabel("True Note")
ax.set_title("KDLY Retrained MLP — Confusion Matrix")
plt.tight_layout()
plt.savefig("kdly_confusion_matrix.png", dpi=150)
print("Saved confusion matrix → kdly_confusion_matrix.png")
plt.show()