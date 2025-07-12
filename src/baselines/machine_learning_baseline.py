import os
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from nilearn.datasets import fetch_atlas_aal
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import skew, kurtosis, entropy
import pywt
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, roc_auc_score

# load data
PREPROC_DIR = "data/fMRI_nii_preprocessed_trial"
LABEL_CSV = "data/labels/aml_dataset_labels_info.csv"
OUTPUT_DIR = "data/mlp_inputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(LABEL_CSV, "r", encoding="utf-8") as f:
    lines = f.readlines()

header = lines[0].strip().replace('"', '').split(",")
rows = [line.strip().replace('"', '').split(",") for line in lines[1:]]
label_df = pd.DataFrame(rows, columns=header)
label_df = label_df[["Image Data ID", "Group"]].dropna()
label_map = {"CN": 0, "LMCI": 1, "AD": 2}
label_df["label"] = label_df["Group"].map(label_map)
label_df = label_df[label_df["label"] != 1]
label_dict = dict(zip(label_df["Image Data ID"].astype(str), label_df["label"]))

atlas = fetch_atlas_aal()
masker = NiftiLabelsMasker(labels_img=atlas.maps, standardize=True, detrend=True, t_r=3.0)

#data augmentation functions
def jitter_signal(signal, sigma=0.005):
    noise = np.random.normal(loc=0, scale=sigma, size=signal.shape)
    return signal + noise

def amplitude_scale(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return signal * scale
from scipy.interpolate import interp1d

def time_warp(signal, sigma=0.05):
    orig_steps = np.arange(signal.shape[0])
    warp_factor = np.random.normal(1.0, sigma)
    new_steps = np.linspace(0, signal.shape[0]-1, int(signal.shape[0] * warp_factor))
    f = interp1d(orig_steps, signal, axis=0, fill_value="extrapolate")
    warped = f(new_steps)
    if warped.shape[0] > signal.shape[0]:
        return warped[:signal.shape[0], :]
    else:
        pad = np.zeros((signal.shape[0] - warped.shape[0], signal.shape[1]))
        return np.vstack([warped, pad])
def random_crop(ts, crop_ratio=0.8):
    T = ts.shape[0]
    crop_len = int(T * crop_ratio)
    start = np.random.randint(0, T - crop_len)
    cropped = ts[start:start + crop_len, :]
    pad_len = T - crop_len
    if pad_len > 0:
        pad = np.tile(cropped[-1:], (pad_len, 1))
        return np.vstack([cropped, pad])
    return cropped


augmentations_pool = [jitter_signal, amplitude_scale, time_warp, random_crop]

#handcrafted features 
def wavelet_entropy(ts, wavelet='db4', level=4):
    coeffs = pywt.wavedec(ts, wavelet, level=level)
    energy = np.array([np.sum(c**2) for c in coeffs])
    energy /= np.sum(energy) + 1e-8
    return -np.sum(energy * np.log2(energy + 1e-8))

def hurst_exponent(ts):
    N = len(ts)
    T = np.arange(1, N + 1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.max(Y) - np.min(Y)
    S = np.std(ts)
    if S == 0: return 0.5
    return np.log(R / S) / np.log(N)

def teager_kaiser_energy(ts):
    tkeo = np.zeros_like(ts)
    tkeo[1:-1] = ts[1:-1]**2 - ts[:-2] * ts[2:]
    return np.mean(tkeo)

#build features
X, y, subjects = [], [], []

for fname in tqdm(os.listdir(PREPROC_DIR), desc="Extracting features"):
    if not fname.endswith(".nii.gz"):
        continue
    subject_id = fname.split("_")[0]
    if subject_id not in label_dict:
        continue
    img_path = os.path.join(PREPROC_DIR, fname)
    try:
        time_series = masker.fit_transform(img_path)  # shape (T, n_ROIs)
    
        n_augs = 3 if label_dict[subject_id] == 2 else 3  # augmentations

        for aug_idx in range(n_augs + 1):
            ts_aug = time_series.copy()
            if aug_idx > 0:
                #we choose a random set of augmentation functions 
                ops = np.random.choice(augmentations_pool, size=np.random.randint(1, 3), replace=False)
                for op in ops:
                    ts_aug = op(ts_aug)

            features = []
            for i in range(ts_aug.shape[1]):
                ts = ts_aug[:, i]
                hist = np.histogram(ts, bins=20, density=True)[0] + 1e-8
                features.extend([
                    np.mean(ts),
                    np.std(ts),
                    skew(ts),
                    kurtosis(ts),
                    entropy(hist),
                    hurst_exponent(ts),
                    teager_kaiser_energy(ts),
                    wavelet_entropy(ts)
                ])

            if len(features) != 928: #sanity check
                print(f"❌ Skipping {fname} (aug{aug_idx}): expected 928 features, got {len(features)}")
                continue

            X.append(features)
            y.append(label_dict[subject_id])
            subjects.append(f"{subject_id}_aug{aug_idx}" if aug_idx > 0 else subject_id)


    except Exception as e:
        print(f"⚠️ Error in {fname}: {e}")

X = np.array(X)
y = np.array(y)
subjects = np.array(subjects)
base_subjects = np.array([s.split("_")[0] for s in subjects])  # we will use data augmentation only in the training set

# --- Baseline MLP ---
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(X, y, groups=base_subjects))

all_f1s, all_precisions, all_recalls, all_accuracies, all_aurocs = [], [], [], [], []
all_train_losses, all_val_losses, all_val_f1s = [], [], []

for fold, (train_idx, test_idx) in enumerate(folds):
    print(f"\n========== Fold {fold + 1} ==========")

    X_train_all, X_test = X[train_idx], X[test_idx]
    y_train_all, y_test = y[train_idx], y[test_idx]

    s_train_idx, s_val_idx = train_test_split(
        np.arange(len(X_train_all)), test_size=0.2, stratify=y_train_all, random_state=42
    )
    X_train, y_train = X_train_all[s_train_idx], y_train_all[s_train_idx]
    X_val, y_val = X_train_all[s_val_idx], y_train_all[s_val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) #only fit_transform in train data!
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                        solver='adam', max_iter=1, warm_start=True, random_state=42)

    best_val_f1 = -1
    patience, counter = 10, 0
    train_losses, val_losses, val_f1s = [], [], []

    for epoch in range(1, 201):
        mlp.fit(X_train, y_train)
        train_loss = 1 - mlp.score(X_train, y_train)
        val_loss = 1 - mlp.score(X_val, y_val)
        y_pred_val = mlp.predict(X_val)
        f1 = f1_score(y_val, y_pred_val, average='macro', zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_f1s.append(f1)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                                       solver='adam', max_iter=1, warm_start=True, random_state=42)
            best_model.__dict__ = mlp.__dict__.copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping (val_f1)")
                break

    y_pred_test = best_model.predict(X_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro', zero_division=0)
    prec_test = precision_score(y_test, y_pred_test, average='macro', zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, average='macro', zero_division=0)
    acc_test = np.mean(y_test == y_pred_test)

    all_f1s.append(f1_test)
    all_precisions.append(prec_test)
    all_recalls.append(rec_test)
    all_accuracies.append(acc_test)
    all_train_losses.append(train_losses)
    all_val_losses.append(val_losses)
    all_val_f1s.append(val_f1s)
    y_test_bin = (y_test == 2).astype(int)
    if hasattr(best_model, "predict_proba"):
        y_score = best_model.predict_proba(X_test)[:, 1] if 2 in best_model.classes_ else np.zeros_like(y_test_bin)
    else:
        y_score = np.zeros_like(y_test_bin)

    try:
        auroc = roc_auc_score(y_test_bin, y_score)
    except ValueError:
        auroc = np.nan

    all_aurocs.append(auroc)
    print(classification_report(y_test, y_pred_test, target_names=["CN", "AD"], digits=4, zero_division=0))

# Print results
print("\n📊 FINAL RESULTS (5-Fold CV):")
print(f"   F1-score (macro)     : {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
print(f"   Precision (macro)    : {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
print(f"   Recall (macro)       : {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
print(f"   Accuracy             : {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"   AUROC                : {np.nanmean(all_aurocs):.4f} ± {np.nanstd(all_aurocs):.4f}")

# Plots
os.makedirs("outputs", exist_ok=True)
max_epochs = max(len(l) for l in all_val_losses)
pad = lambda x: x + [np.nan]*(max_epochs - len(x))

mean_train = np.nanmean([pad(l) for l in all_train_losses], axis=0)
mean_val = np.nanmean([pad(l) for l in all_val_losses], axis=0)
mean_f1 = np.nanmean([pad(l) for l in all_val_f1s], axis=0)

plt.figure(figsize=(10, 5))
plt.plot(mean_train, label="Train Loss")
plt.plot(mean_val, label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/mlp_loss_curve.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(mean_f1, label="Validation F1 (macro)", color="purple")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.title("Validation F1 Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/mlp_f1_curve.png")
plt.close()