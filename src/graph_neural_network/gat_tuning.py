import itertools
import pickle
import torch
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score
from torch_geometric.loader import DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import GroupKFold, train_test_split
from src.graph_neural_network.graph_attention_network import GATClassifier, select_graphs, evaluate, train, compute_val_loss
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

GRAPH_PATH = "data/pyg_graphs/graphs.pk_linear"
METADATA_PATH = "data/pyg_graphs/graph_metadata_linear.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(GRAPH_PATH, "rb") as f:
    data_list = pickle.load(f)

metadata_df = pd.read_csv(METADATA_PATH)

for data in data_list:
    if data.y.item() == 2:
        data.y = torch.tensor([1], dtype=torch.long)

metadata_df["is_augmented"] = metadata_df["subject_id"].str.contains("_aug")
metadata_df["subject_real"] = metadata_df["subject_id"].str.replace(r"_aug\\d*", "", regex=True)
data_list = [data for data in data_list if data.edge_index.max().item() < data.x.size(0)]

label_names = {0: "CN", 1: "AD"}
in_dim = data_list[0].num_node_features
num_classes = len(label_names)

real_subjects = metadata_df[~metadata_df["is_augmented"]].copy()
subject_df = real_subjects.groupby("subject_real").first().reset_index()
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(subject_df, subject_df["label"], groups=subject_df["subject_real"]))


param_grid = {
    "hidden_channels": [32, 64, 128],
    "lr": [1e-2, 1e-3, 5e-4, 1e-4],
    "weight_decay": [0.0, 1e-4, 1e-5],
    "dropout": [0.3, 0.5, 0.6],
    "heads": [2, 4, 8]
}

results = []

for params in itertools.product(*param_grid.values()):
    hparams = dict(zip(param_grid.keys(), params))
    print(f"\n Testing combination: {hparams}")
    
    fold_scores = []

    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(folds):
        train_subj = subject_df.iloc[train_idx]["subject_real"].tolist()
        test_subj = subject_df.iloc[test_idx]["subject_real"].tolist()

        stratify_labels = subject_df.iloc[train_idx]["label"].values
        s_train_idx, s_val_idx = train_test_split(
            np.arange(len(train_subj)),
            test_size=0.2,
            stratify=stratify_labels,
            random_state=42
        )
        train_fold_subj = [train_subj[i] for i in s_train_idx]
        val_fold_subj = [train_subj[i] for i in s_val_idx]

        data_train = select_graphs(train_fold_subj, allow_augmented=True)
        data_val = select_graphs(val_fold_subj, allow_augmented=False)

        y_train = [d.y.item() for d in data_train]
        class_sample_count = Counter(y_train)
        weights = [1.0 / class_sample_count[y] for y in y_train]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_loader = DataLoader(data_train, batch_size=16, sampler=sampler)
        val_loader = DataLoader(data_val, batch_size=16)

        class_weights = torch.tensor([
            len(y_train) / (2 * class_sample_count[i]) for i in label_names
        ]).to(device)

        model = GATClassifier(
            in_channels=in_dim,
            hidden_channels=hparams["hidden_channels"],
            out_channels=num_classes,
            dropout=hparams["dropout"]
        ).to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )

        best_f1, best_state = -1, None
        patience, counter = 10, 0

        for epoch in range(1, 201):
            train(model, train_loader, optimizer, class_weights)
            val_loss = compute_val_loss(model, val_loader, class_weights)
            y_true, y_pred, y_probs = evaluate(model, val_loader, return_probs=True)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_state = model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        model.load_state_dict(best_state)
        y_true, y_pred, y_probs = evaluate(model, val_loader, return_probs=True)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            auroc = roc_auc_score(y_true, y_probs)
        except:
            auroc = float('nan')

        fold_metrics.append({
            "f1": f1,
            "accuracy": acc,
            "recall": recall,
            "auroc": auroc
        })

    avg_metrics = {
        k: np.nanmean([m[k] for m in fold_metrics]) for k in fold_metrics[0]
    }
    std_metrics = {
        k: np.nanstd([m[k] for m in fold_metrics]) for k in fold_metrics[0]
    }

    print(f"\n Avg Val Metrics over folds:")
    for k in avg_metrics:
        print(f"   - {k.upper()} : {avg_metrics[k]:.4f} ± {std_metrics[k]:.4f}")

    hparams.update({
        f"avg_{k}": avg_metrics[k] for k in avg_metrics
    })
    hparams.update({
        f"std_{k}": std_metrics[k] for k in std_metrics
    })
    results.append(hparams)

