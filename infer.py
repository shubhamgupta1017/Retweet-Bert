import torch
import argparse
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from tqdm import tqdm

# ---------- CONFIG ----------
parser = argparse.ArgumentParser(description="Train user classifier based on profile embeddings.")
parser.add_argument("--proj_type", type=str, required=True, help="Project folder name (e.g., 'paper_imp')")
args = parser.parse_args()
proj_type = args.proj_type
INPUT_CSV = "data/test.csv"        # input file with users (and maybe labels)
USER_BIN_CSV = "output/users_bin.csv"              # fallback for missing profiles
OUTPUT_CSV = f"preds/{proj_type}_users_predicted.csv"          # output file
MODEL_PATH = f"{proj_type}/rbert"               # SentenceTransformer model
CLASSIFIER_PATH = f"{proj_type}/user_classifier.pt"  # trained classifier checkpoint

# ---------- DEVICE ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- CLASSIFIER DEFINITION ----------
class UserClassifier(torch.nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.3):
        super(UserClassifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.act1 = torch.nn.LeakyReLU(0.1)

        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.act2 = torch.nn.LeakyReLU(0.1)

        self.fc3 = torch.nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim // 4)
        self.act3 = torch.nn.LeakyReLU(0.1)

        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(hidden_dim // 4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.act1(self.bn1(self.fc1(x)))
        x2 = self.act2(self.bn2(self.fc2(x1)))
        x3 = self.act3(self.bn3(self.fc3(x2)))
        x = self.dropout(x3)
        x = self.sigmoid(self.out(x))
        return x

# ---------- LOAD MODELS ----------
print("Loading embedding model...")
embed_model = SentenceTransformer(MODEL_PATH)

print("Loading classifier...")
classifier = UserClassifier(input_dim=768, hidden_dim=256, dropout=0.3).to(device)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier.eval()
print("✅ Classifier loaded successfully")

# ---------- LOAD INPUT ----------
print("Loading test data...")
test_df = pd.read_csv(INPUT_CSV)
user_bin = pd.read_csv(USER_BIN_CSV)

# Merge with user_bin to ensure every user has a profile
if 'profile' not in test_df.columns:
    test_df = test_df.merge(user_bin[['user', 'profile']], on='user', how='left')
else:
    test_df = test_df.merge(user_bin[['user', 'profile']], on='user', how='left', suffixes=('', '_bin'))
    test_df['profile'] = test_df['profile'].fillna(test_df['profile_bin'])
    test_df.drop(columns=['profile_bin'], inplace=True)

missing_profiles = test_df['profile'].isna().sum()
if missing_profiles > 0:
    print(f"⚠️ Dropping {missing_profiles} users with no profile text")
    test_df = test_df.dropna(subset=['profile'])

print(f"Total users for prediction: {len(test_df)}")

# ---------- GENERATE EMBEDDINGS ----------
print("Encoding profiles...")
embeddings = embed_model.encode(
    test_df['profile'].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=32
)

# ---------- PREDICTION ----------
print("Predicting labels...")
X = torch.tensor(embeddings, dtype=torch.float32).to(device)

with torch.no_grad():
    preds = classifier(X).cpu().numpy().flatten()

labels = (preds > 0.5).astype(int)

# ---------- SAVE OUTPUT ----------
output_df = pd.DataFrame({
    'user': test_df['user'],
    'pred': labels
})

output_df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Predictions saved to {OUTPUT_CSV}")
