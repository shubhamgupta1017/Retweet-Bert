import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


# === Parse command-line arguments ===
parser = argparse.ArgumentParser(description="Train user classifier based on profile embeddings.")
parser.add_argument("--proj_type", type=str, required=True, help="Project folder name (e.g., 'paper_imp')")
args = parser.parse_args()
proj_type = args.proj_type


# === Config ===
EPOCHS = 100
OUTPUT_MODEL_PATH = f"{proj_type}/user_classifier.pt"
MODEL_PATH = f"./{proj_type}/rbert"
TRAIN_FILE = "data/test.csv"
USER_BIN_FILE = "output/users_bin.csv"
LR = 1e-4
DROPOUT = 0.3
HIDDEN_DIM = 256
TEST_SIZE = 0.2
RANDOM_STATE = 42


# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# === Load model ===
model_st = SentenceTransformer(MODEL_PATH)


# === Load data ===
train_df = pd.read_csv(TRAIN_FILE)
users_bin = pd.read_csv(USER_BIN_FILE)

# Merge profiles into full dataframe
full_df = train_df.merge(users_bin, on="user", how="left")

# Drop missing profiles if any
full_df = full_df.dropna(subset=["profile"])
print("Final dataset size:", full_df.shape)


# === Train/Test Split ===
train_df, test_df = train_test_split(
    full_df, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=full_df["label"]
)

print("Data shapes → Train:", train_df.shape, "| Test:", test_df.shape)


# === Encode embeddings ===
print("Encoding train embeddings...")
train_embeddings = model_st.encode(train_df["profile"].tolist(), show_progress_bar=True)

print("Encoding test embeddings...")
test_embeddings = model_st.encode(test_df["profile"].tolist(), show_progress_bar=True)

X_train = np.array(train_embeddings)
y_train = train_df["label"].astype(float).values
X_val = np.array(test_embeddings)
y_val = test_df["label"].astype(float).values

print("Train embeddings shape:", X_train.shape)
print("Test embeddings shape:", X_val.shape)


# === Convert to tensors ===
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)


# === Define classifier ===
class UserClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super(UserClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = nn.BatchNorm1d(hidden_dim // 4)
        self.act3 = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim // 4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.act1(self.bn1(self.fc1(x)))
        x2 = self.act2(self.bn2(self.fc2(x1)))
        x3 = self.act3(self.bn3(self.fc3(x2)))
        x = self.dropout(x3)
        x = self.sigmoid(self.out(x))
        return x


# === Initialize model ===
model = UserClassifier(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)


# === Training loop ===
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)
        val_acc = ((val_pred > 0.5).float() == y_val).float().mean()

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train Loss: {loss.item():.4f} | "
          f"Val Loss: {val_loss.item():.4f} | "
          f"Val Acc: {val_acc.item():.4f}")


# === Save model ===
torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
print(f"Model saved to {OUTPUT_MODEL_PATH}")
