import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# === Load CSVs ===
parser = argparse.ArgumentParser(description="Train user classifier based on profile embeddings.")
parser.add_argument("--proj_type", type=str, required=True, help="Project folder name (e.g., 'paper_imp')")
args = parser.parse_args()
proj_type = args.proj_type
truth_df = pd.read_csv("data/test.csv")
pred_df = pd.read_csv(f"preds/{proj_type}_users_predicted.csv")
merged_output_path=f"preds/{proj_type}_merged_data.csv"
# === Merge on 'user' ===
df = truth_df.merge(pred_df, on="user", how="inner")

# Ensure numeric
df["label"] = df["label"].astype(float)
df["pred"] = df["pred"].astype(float)

# === If 'pred' are probabilities, threshold at 0.5 ===
df["pred_label"] = (df["pred"] >= 0.5).astype(int)

# === Compute Metrics ===
accuracy = accuracy_score(df["label"], df["pred_label"])

# Safe AUC (works only if both classes are present)
try:
    auc = roc_auc_score(df["label"], df["pred"])
except ValueError:
    auc = None  # If AUC cannot be computed

macro_f1 = f1_score(df["label"], df["pred_label"], average="macro")

# === Print results ===
print("=== Evaluation Metrics ===")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Macro F1  : {macro_f1:.4f}")
print(f"AUC       : {auc:.4f}" if auc is not None else "AUC: Not applicable (single-class)")

df.to_csv(merged_output_path, index=False)
print(f"\nSaved {merged_output_path} for inspection.")
