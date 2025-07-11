import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from nilearn.datasets import fetch_atlas_aal
from nilearn.maskers import NiftiLabelsMasker
from scipy.stats import skew, kurtosis, entropy
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
import pywt

PREPROC_DIR = "data/fMRI_nii_preprocessed_trial"
LABEL_CSV = "data/labels/aml_dataset_labels_info.csv"

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
    return np.log(R / S) / np.log(N) if S != 0 else 0.5

def teager_kaiser_energy(ts):
    tkeo = np.zeros_like(ts)
    tkeo[1:-1] = ts[1:-1]**2 - ts[:-2] * ts[2:]
    return np.mean(tkeo)
 
X, y, subjects = [], [], []

for fname in tqdm(os.listdir(PREPROC_DIR), desc="Extracting features"):
    if not fname.endswith(".nii.gz"):
        continue
    subject_id = fname.split("_")[0]
    if subject_id not in label_dict:
        continue
    img_path = os.path.join(PREPROC_DIR, fname)
    try:
        time_series = masker.fit_transform(img_path)

        n_augs = 3 if label_dict[subject_id] == 2 else 1

        for aug_idx in range(n_augs + 1):
            ts_aug = time_series.copy()
            if aug_idx > 0:
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

            if len(features) != 928:
                continue

            X.append(features)
            y.append(label_dict[subject_id])
            subjects.append(f"{subject_id}_aug{aug_idx}" if aug_idx > 0 else subject_id)

    except Exception as e:
        print(f"⚠️ Error in {fname}: {e}")

X = np.array(X)
y = np.array(y)
subjects = np.array(subjects)

gkf = GroupKFold(n_splits=5)
subject_base_ids = np.array([s.split("_")[0] for s in subjects])
folds = list(gkf.split(X, y, groups=subject_base_ids))

all_f1s, all_precisions, all_recalls, all_accuracies, all_aurocs = [], [], [], [], []

for fold, (train_idx, test_idx) in enumerate(folds):
    print(f"\n========== Fold {fold + 1} ==========")
    y_train = y[train_idx]
    y_test = y[test_idx]

    classes, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum()

    y_pred = np.random.choice(classes, size=len(y_test), p=probs)

    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_test, y_pred)


    y_true_binary = (y_test == 2).astype(int)
    ad_prob = probs[classes == 2][0] if 2 in classes else 0.0
    y_score = np.full_like(y_test, fill_value=ad_prob, dtype=np.float64)

    try:
        auroc = roc_auc_score(y_true_binary, y_score)
    except ValueError:
        auroc = np.nan

    all_f1s.append(f1)
    all_precisions.append(prec)
    all_recalls.append(rec)
    all_accuracies.append(acc)
    all_aurocs.append(auroc)


    print(f"Fold {fold + 1} Results:")
    print(f"  F1-score (macro): {f1:.4f}")
    print(f"  Precision       : {prec:.4f}")
    print(f"  Recall          : {rec:.4f}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  AUROC (CN vs AD): {auroc:.4f}")


print("\n RANDOM BASELINE (5-Fold CV):")
print(f"   F1-score (macro)     : {np.mean(all_f1s):.4f} ± {np.std(all_f1s):.4f}")
print(f"   Precision (macro)    : {np.mean(all_precisions):.4f} ± {np.std(all_precisions):.4f}")
print(f"   Recall (macro)       : {np.mean(all_recalls):.4f} ± {np.std(all_recalls):.4f}")
print(f"   Accuracy             : {np.mean(all_accuracies):.4f} ± {np.std(all_accuracies):.4f}")
print(f"   AUROC                : {np.nanmean(all_aurocs):.4f} ± {np.nanstd(all_aurocs):.4f}")
