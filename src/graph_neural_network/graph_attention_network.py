import os
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_max_pool, BatchNorm, global_mean_pool
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from collections import Counter
import numpy as np
import pandas as pd
from torch.utils.data import WeightedRandomSampler
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*torch-scatter.*")

# PATHS
GRAPH_PATH = "data/pyg_graphs/graphs.pk_linear"
METADATA_PATH = "data/pyg_graphs/graph_metadata_linear.csv"
LOGS_DIR = "data/results"
os.makedirs(LOGS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using:", device)
if device.type == 'cuda':
    print("GPU:", torch.cuda.get_device_name(0))

# Load data
with open(GRAPH_PATH, "rb") as f:
    data_list = pickle.load(f)

metadata_df = pd.read_csv(METADATA_PATH)

for data in data_list:
    if data.y.item() == 2:
        data.y = torch.tensor([1], dtype=torch.long)


# find augmented data
metadata_df["is_augmented"] = metadata_df["subject_id"].str.contains("_aug")
metadata_df["subject_real"] = metadata_df["subject_id"].str.replace(r"_aug\\d*", "", regex=True)

# Filter only valid graphs
print("Checking edge_index integrity...")
data_list = [data for data in data_list if data.edge_index.max().item() < data.x.size(0)]

# Global variables

label_names = {0: "CN", 1: "AD"}
in_dim = data_list[0].num_node_features
num_classes = len(label_names)

# Grouping real subjects
real_subjects = metadata_df[~metadata_df["is_augmented"]].copy()
subject_df = real_subjects.groupby("subject_real").first().reset_index()

# GroupKFold
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(subject_df, subject_df["label"], groups=subject_df["subject_real"]))

def select_graphs(subject_list, allow_augmented):
    sel_ids = metadata_df[
        metadata_df["subject_real"].isin(subject_list) &
        ((~metadata_df["is_augmented"]) | allow_augmented)
    ].index.tolist()
    return [data_list[i] for i in sel_ids]

# GAT model
# we pass the embeddings only throughout the linear layer
class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        #we reduce dimensionality of the CNN embeddings from 512 to 16
        self.cnn_bn = nn.BatchNorm1d(512)
        self.cnn_linear = nn.Linear(512, 16)

        self.keep_features_and_asym = in_channels - 512
        print(f"Handcrafted + Asymmetry feature dim: {self.keep_features_and_asym}") #this dimension is also 16

        self.feat_norm = nn.LayerNorm(self.keep_features_and_asym)

        self.pre = nn.Sequential(
            nn.BatchNorm1d(16 + self.keep_features_and_asym),
            nn.Linear(16 + self.keep_features_and_asym, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout) 
        )
        #we define  the GAT convolutional layer
        self.gat1 = GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout)
        self.norm1 = BatchNorm(hidden_channels * heads)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),  
            torch.nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x_cnn = x[:, :512]
        x_feat_asym = self.feat_norm(x[:, 512:])
        x_cnn = self.cnn_bn(x_cnn)
        x_cnn = self.cnn_linear(x_cnn)
        #concatenate reduced embedding features with handcrafted + asymetry features 
        x = torch.cat([x_cnn, x_feat_asym], dim=1)

        x = self.pre(x)
        x = self.gat1(x, edge_index)
        x = self.norm1(x)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

# Train function
def train(model, loader, optimizer, class_weights):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y, weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def compute_val_loss(model, loader, class_weights):
    model.eval()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        loss = F.cross_entropy(out, data.y, weight=class_weights)
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, return_probs=False):
    model.eval()
    all_preds, all_labels = [], []
    all_probs = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        preds = out.argmax(dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(data.y.cpu())
        if return_probs:
            probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
    if return_probs:
        return all_labels, all_preds, all_probs
    return all_labels, all_preds

# Cross-validation
all_f1s, all_precisions, all_recalls, all_accuracies, all_aurocs  = [], [], [], [], []

all_train_losses = []
all_val_losses = []
all_val_f1s = []

for fold, (train_subject_idx, test_subject_idx) in enumerate(folds):
    train_losses = []
    val_losses = []
    val_f1s = []
    print(f"\n========== Fold {fold + 1} ==========")
    train_subjects = subject_df.iloc[train_subject_idx]["subject_real"].tolist()
    test_subjects = subject_df.iloc[test_subject_idx]["subject_real"].tolist()

    train_subjects_df = subject_df.iloc[train_subject_idx]
    stratify_labels = train_subjects_df["label"].values
    s_train_idx, s_val_idx = train_test_split(
        np.arange(len(train_subjects_df)),
        test_size=0.2,
        stratify=stratify_labels,
        random_state=42
    )
    train_subj = train_subjects_df.iloc[s_train_idx]["subject_real"].tolist()
    val_subj = train_subjects_df.iloc[s_val_idx]["subject_real"].tolist()

    data_train = select_graphs(train_subj, allow_augmented=True)
    data_val = select_graphs(val_subj, allow_augmented=False)
    data_test = select_graphs(test_subjects, allow_augmented=False)

    y_train = [d.y.item() for d in data_train]
    class_sample_count = Counter(y_train)
    weights = [1.0 / class_sample_count[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(data_train, batch_size=16, sampler=sampler)
    val_loader = DataLoader(data_val, batch_size=16)
    test_loader = DataLoader(data_test, batch_size=16)

    class_weights = torch.tensor([
        len(y_train) / (2 * class_sample_count[i]) for i in label_names
    ]).to(device)
    model = GATClassifier(
        in_channels=in_dim,
        hidden_channels=64,
        out_channels=num_classes,
        dropout=0.5  
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    best_val_f1 = -1
    patience, counter = 10, 0

    for epoch in range(1, 201):
        train_loss = train(model, train_loader, optimizer, class_weights)
        val_loss = compute_val_loss(model, val_loader, class_weights)
        y_true_val, y_pred_val = evaluate(model, val_loader)
        f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(f1)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model_state = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping (val_f1)")
                break


    # Evaluate on test
    model.load_state_dict(best_model_state)
    y_true_test, y_pred_test, y_probs_test = evaluate(model, test_loader, return_probs=True)
    f1_test = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    prec_test = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    rec_test = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    acc_test = np.mean(np.array(y_true_test) == np.array(y_pred_test))
    try:
        auroc = roc_auc_score(y_true_test, y_probs_test)
    except ValueError:
        auroc = float('nan')
    all_f1s.append(f1_test)
    all_precisions.append(prec_test)
    all_recalls.append(rec_test)
    all_accuracies.append(acc_test)
    all_aurocs.append(auroc)

    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_val_f1s.append(val_f1s)

    print(classification_report(y_true_test, y_pred_test, target_names=["CN", "AD"], digits=4, zero_division=0))


# Final report
print("\n📊 FINAL RESULT (5-Fold CV Test Set):")
print(f"   F1-score (macro)     : {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
print(f"   Precision (macro)    : {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
print(f"   Recall (macro)       : {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
print(f"   Accuracy             : {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"   AUROC (class AD)     : {np.nanmean(all_aurocs):.4f} ± {np.nanstd(all_aurocs):.4f}")

max_epochs = max(len(losses) for losses in all_val_losses)

def pad_list(lst, target_len):
    return lst + [np.nan] * (target_len - len(lst))

train_losses_padded = np.array([pad_list(l, max_epochs) for l in all_train_losses])
val_losses_padded = np.array([pad_list(l, max_epochs) for l in all_val_losses])
val_f1s_padded = np.array([pad_list(l, max_epochs) for l in all_val_f1s])

mean_train = np.nanmean(train_losses_padded, axis=0)
std_train = np.nanstd(train_losses_padded, axis=0)

mean_val = np.nanmean(val_losses_padded, axis=0)
std_val = np.nanstd(val_losses_padded, axis=0)

mean_f1 = np.nanmean(val_f1s_padded, axis=0)
std_f1 = np.nanstd(val_f1s_padded, axis=0)

epochs = np.arange(1, max_epochs + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, mean_train, label="Train Loss", linewidth=2)
plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.3)

plt.plot(epochs, mean_val, label="Validation Loss", linewidth=2)
plt.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.3)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Average Train vs Validation Loss (5-Fold CV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{LOGS_DIR}/loss_curve_average.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(epochs, mean_f1, label="Validation F1 (macro)", color="purple", linewidth=2)
plt.fill_between(epochs, mean_f1 - std_f1, mean_f1 + std_f1, alpha=0.3, color="purple")

plt.xlabel("Epoch")
plt.ylabel("F1 Score (Macro)")
plt.title("Average Validation F1 Score (5-Fold CV)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{LOGS_DIR}/f1_curve_average.png")
plt.close()