import os
import numpy as np
import pandas as pd
import nibabel as nib
import pywt
import pickle
from tqdm import tqdm
import networkx as nx
from nilearn.datasets import fetch_atlas_aal
from nilearn.maskers import NiftiLabelsMasker
from torch_geometric.data import Data
from torchvision.models.video import r3d_18
from torch import nn
from scipy.interpolate import interp1d
from scipy.ndimage import zoom
import torch
from torch.nn.functional import adaptive_max_pool3d
from scipy.stats import skew, kurtosis, entropy
from scipy.fftpack import fft
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from nilearn.image import resample_to_img
import traceback

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU available:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU Found")


atlas = fetch_atlas_aal()
atlas_img = nib.load(atlas.maps) 
atlas_img = atlas.maps


labels = atlas.labels
indices = atlas.indices

keywords = ["Hippocampus", "Parahippocampal", "Precuneus", "Cingulate",   # ROIs names important in Alzheimer
            "Temporal", "Entorhinal", "Insula", "Frontal_Medial", "Pole"]

relevant_rois = [(idx, label) for idx, label in zip(indices, labels)
                 if any(k in label for k in keywords)]


for idx, label in relevant_rois:
    print(f"{idx}: {label}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


PREPROC_DIR = "data/fMRI_nii_preprocessed_trial"
LABEL_CSV = "data/labels/aml_dataset_labels_info.csv"
OUTPUT_DIR = "data/pyg_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


TR = 3.0
EMB_DIM = 512 


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
print("Labels loaded:", len(label_dict))



cnn3d = r3d_18(pretrained=True)
cnn3d.fc = nn.Identity()
cnn3d = cnn3d.to(device)
cnn3d.eval()

RELEVANT_ROIS = [
    3001, 3002,       # Insula
    4101, 4102,       # Hippocampus
    6301, 6302,       # Precuneus
    8111, 8112,       # Superior Temporal Gyrus
    8121, 8122,       # Temporal Pole: Superior Temporal Gyrus
    8201, 8202,       # Middle Temporal Gyrus
    8211, 8212,       # Temporal Pole: Middle Temporal Gyrus
    8301, 8302        # Inferior Temporal Gyrus
]

TARGET_SHAPE = (12, 12, 12)


def parallel_apply(func, data):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(func, data))
    return np.array(results)


def hurst_exponent(ts):
    N = len(ts)
    T = np.arange(1, N + 1)
    Y = np.cumsum(ts - np.mean(ts))
    R = np.max(Y) - np.min(Y)
    S = np.std(ts)
    if S == 0:
        return 0.5
    return np.log(R / S) / np.log(N)

def teager_kaiser_energy(ts):
    tkeo = np.zeros_like(ts)
    tkeo[1:-1] = ts[1:-1]**2 - ts[:-2] * ts[2:]
    return np.mean(tkeo)

def wavelet_entropy(ts, wavelet='db4', level=4):
    coeffs = pywt.wavedec(ts, wavelet, level=level)
    energy = np.array([np.sum(c**2) for c in coeffs])
    energy /= np.sum(energy) + 1e-8
    return -np.sum(energy * np.log2(energy + 1e-8))


def extract_roi_volumes_batch(img_data, roi_mask, roi_values):
    vols = []
    T = img_data.shape[3]
    time = np.arange(T)
    for i, roi_value in enumerate(roi_values):
        
        assert roi_mask.ndim == 3, f"roi_mask is not 3D, shape={roi_mask.shape}"
        roi_binary = (roi_mask == roi_value)

        if roi_binary.ndim != 3:
            raise ValueError(f"roi_binary is not in 3D, shape={roi_binary.shape}")

        if not np.any(roi_binary):
            print(f"ROI {roi_value} is empty, 0 volume will be added")
            vols.append(np.zeros((3, *TARGET_SHAPE), dtype=np.float32))
            continue

        try:
            roi_4d = img_data * roi_binary[..., np.newaxis]  # [X,Y,Z,T]
        except Exception as e:
            print(f"Error when multiplying img_data * roi_binary[..., np.newaxis]: {e}")
            raise

        mean_vol = np.mean(roi_4d, axis=3)
        std_vol = np.std(roi_4d, axis=3)

        slope_vol = np.zeros_like(mean_vol)

        assert roi_binary.ndim == 3, f"roi_binary has unexpected shape: {roi_binary.shape}"
        assert roi_binary.shape == img_data.shape[:3], \
            f"Mismatch: roi_binary.shape={roi_binary.shape}, img_data.shape[:3]={img_data.shape[:3]}"
        
        flat_mask = roi_binary.flatten()
        reshaped = roi_4d.reshape(-1, T)

        if np.any(flat_mask):
            valid_voxels = reshaped[flat_mask]
            slopes = np.array([np.polyfit(time, voxel_ts, 1)[0] for voxel_ts in valid_voxels])
            slope_vol.flat[flat_mask] = slopes

        vol = np.stack([mean_vol, std_vol, slope_vol], axis=0)
        zoom_factors = [1.0] + [t / s for s, t in zip(vol.shape[1:], TARGET_SHAPE)]
        vol_resized = zoom(vol, zoom_factors, order=1).astype(np.float32)
        vols.append(vol_resized)

    return np.stack(vols)  # [N, 3, D, H, W]


def extract_embeddings_batch(volumes):
    vols_tensor = torch.tensor(volumes, dtype=torch.float32).to(device)
    with torch.no_grad():
        x = cnn3d.stem(vols_tensor)
        x = cnn3d.layer1(x)
        x = cnn3d.layer2(x)
        x = cnn3d.layer3(x)
        x = cnn3d.layer4(x)
        x = adaptive_max_pool3d(x, 1) 
        x = x.view(x.size(0), -1)  # [N, 512]
        embs = x.cpu().numpy()
    return embs

"""
def resize_to_shape(volume, target_shape=(16, 16, 16)):
    Reescala un volumen [C, D, H, W] a [C, target_shape...]
    if volume.ndim != 4 or len(target_shape) != 3:
        raise ValueError("Volume debe ser [C, D, H, W] y target_shape de 3 elementos.")
    
    channels = volume.shape[0]
    zoom_factors = [1.0] + [t / s for s, t in zip(volume.shape[1:], target_shape)]
    return zoom(volume, zoom_factors, order=1).astype(np.float32)
"""

def extract_node_features_with_topology(time_series, adj_matrix, tr=3.0):
    means = np.mean(time_series, axis=0)
    stds = np.std(time_series, axis=0)
    mins = np.min(time_series, axis=0)
    maxs = np.max(time_series, axis=0)
    skews = skew(time_series, axis=0)
    kurt_vals = kurtosis(time_series, axis=0)
    power = np.mean(np.square(time_series), axis=0)
    norm_ts = time_series - np.min(time_series, axis=0)
    norm_ts = norm_ts / (np.sum(norm_ts, axis=0) + 1e-8)
    entrs = entropy(norm_ts, axis=0)
    fft_vals = np.abs(fft(time_series, axis=0))
    fft_dom = fft_vals[1, :]
    n_timepoints = time_series.shape[0]
    freqs = np.fft.fftfreq(n_timepoints, d=tr)
    pos_freqs = freqs > 0
    low_freq_mask = (freqs >= 0.01) & (freqs <= 0.1)
    alff = np.mean(fft_vals[low_freq_mask, :], axis=0)
    falff = alff / (np.sum(fft_vals[pos_freqs, :], axis=0) + 1e-8)
    roi_signals = [time_series[:, i] for i in range(time_series.shape[1])]
    hursts = parallel_apply(hurst_exponent, roi_signals)
    tkeos = parallel_apply(teager_kaiser_energy, roi_signals)
    wavelets = parallel_apply(wavelet_entropy, roi_signals)
    G = nx.from_numpy_array(adj_matrix)
    deg = np.array([val for _, val in G.degree()])
    clustering = np.array(list(nx.clustering(G).values()))
    betweenness = np.array(list(nx.betweenness_centrality(G).values()))
    pagerank = np.array(list(nx.pagerank(G).values()))
    features = np.stack([
        means, stds, mins, maxs,
        skews, kurt_vals, power, entrs, fft_dom,
        hursts, tkeos, wavelets,
        deg, clustering, betweenness,
        alff, falff, pagerank
    ], axis=1)
    return features

# --- Funciones de data augmentation ---
def jitter_signal(signal, sigma=0.005):
    noise = np.random.normal(loc=0, scale=sigma, size=signal.shape)
    return signal + noise


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

def amplitude_scale(ts, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(*scale_range)
    return ts * scale


augmentations_pool = [jitter_signal, amplitude_scale] #time_warp, random_crop

metadata_path = os.path.join(OUTPUT_DIR, "graph_metadata_fix_short.csv")
processed_subjects = set()
if os.path.exists(metadata_path):
    prev_metadata = pd.read_csv(metadata_path)
    processed_subjects = set(prev_metadata["subject_id"].astype(str).str.extract(r"(^[^\_]+)")[0])
    print(f"Subjects already processed: {len(processed_subjects)}")

graphs, metadata, new_files = [], [], []

for fname in os.listdir(PREPROC_DIR):
    if not fname.endswith(".nii.gz"): continue
    subject_id = fname.split("_")[0]
    if subject_id not in label_dict or subject_id in processed_subjects: continue
    new_files.append((subject_id, fname))

lock = Lock()  


def process_subject(subject_id, fname):
    if subject_id not in label_dict or subject_id in processed_subjects:
        return None, None

    label = label_dict[subject_id]
    img_path = os.path.join(PREPROC_DIR, fname)

    try:
        print(f"Processing subject: {subject_id}")
        img = nib.load(img_path)
        img_data = img.get_fdata()

        if img_data.ndim != 4:
            raise ValueError(f"The image is not 4D: shape={img_data.shape}")

        resampled_atlas = resample_to_img(
            atlas_img, img,
            interpolation="nearest",
            force_resample=True,
            copy_header=True
        )

        roi_raw = resampled_atlas.get_fdata()

        if roi_raw.ndim == 4:
            print("Atlas has 4 dim. Extracting 1st volume...")
            roi_raw = roi_raw[..., 0]

        if roi_raw.ndim != 3:
            raise ValueError(f"Resampled ROI does not have 3D: shape={roi_raw.shape}")

        roi_ids = roi_raw.astype(int)

        if roi_ids.shape != img_data.shape[:3]:
            raise ValueError(f"ROI shape {roi_ids.shape} != fMRI shape {img_data.shape[:3]}")

        if np.all(img_data == 0):
            raise ValueError("The image is empty (all nulls)")

        if np.isnan(img_data).any():
            raise ValueError("The image contains NaNs")

        if np.isnan(roi_ids).any():
            raise ValueError("The resampled atlas contains NaNs")
        print("SUCCESS! Image and ATLAS are valid.Proceeding to graph construction.")

        masker = NiftiLabelsMasker(labels_img=resampled_atlas, standardize=False, t_r=TR)
        time_series = masker.fit_transform(img)

        roi_values_all = sorted(v for v in np.unique(roi_ids) if v > 0) 

        assert roi_ids.shape == img_data.shape[:3], f"Error: ROI shape {roi_ids.shape} != fMRI shape {img_data.shape[:3]}"

        subject_graphs = []
        subject_metadata = []


        if label == 2:      # AD augmentations
            n_augs = 3
        elif label == 0:    # CN augmentations
            n_augs = 3
        else:
            n_augs = 0  

        for aug_idx in range(n_augs + 1):
            ts_aug = time_series.copy()
            if aug_idx > 0:
                ops = np.random.choice(augmentations_pool, size=np.random.randint(1, 3), replace=False)
                for op in ops:
                    ts_aug = op(ts_aug)

            corr = np.corrcoef(ts_aug.T)
            np.fill_diagonal(corr, 0)
            z_corr = np.arctanh(corr)
            threshold = np.percentile(np.abs(z_corr), 80)
            adj = (np.abs(z_corr) > threshold).astype(int)

            isolated_mask = np.sum(adj, axis=1) == 0
            if np.all(isolated_mask):
                continue

            ts_aug = ts_aug[:, ~isolated_mask]
            adj = adj[~isolated_mask][:, ~isolated_mask]
            if ts_aug.shape[1] < 5 or np.sum(adj) < 20:
                continue

            valid_roi_values = [roi_values_all[i] for i, valid in enumerate(~isolated_mask) if valid]
            extra_features = extract_node_features_with_topology(ts_aug, adj, tr=TR)

            emb_roi_values = [roi for roi in valid_roi_values if roi in RELEVANT_ROIS]
            if not emb_roi_values:
                continue

            volumes = extract_roi_volumes_batch(img_data, roi_ids, emb_roi_values)
            with lock:
                emb_matrix = extract_embeddings_batch(volumes)

            x_nodes = np.zeros((len(valid_roi_values), EMB_DIM), dtype=np.float32)
            for i, roi in enumerate(emb_roi_values):
                x_nodes[valid_roi_values.index(roi)] = emb_matrix[i]

            atlas_labels_valid = [labels[indices.index(str(i))] for i in valid_roi_values if str(i) in indices]
            label_map = dict(zip(valid_roi_values, atlas_labels_valid))
            asym_by_node = np.zeros_like(extra_features)

            for idx_L, idx_R in zip(*np.where(np.triu(np.ones((len(valid_roi_values), len(valid_roi_values))), 1))):
                name_L = label_map[valid_roi_values[idx_L]]
                name_R = label_map[valid_roi_values[idx_R]]
                if name_L.endswith("_L") and name_L.replace("_L", "_R") == name_R:
                    diff = np.abs(extra_features[idx_L] - extra_features[idx_R])
                    asym_by_node[idx_L] += diff
                    asym_by_node[idx_R] += diff

            x_nodes = np.concatenate([x_nodes, extra_features, asym_by_node], axis=1)

            norms = np.linalg.norm(x_nodes, axis=1, keepdims=True, ord=2)
            x_nodes /= np.where(norms == 0, 1, norms)

            x = torch.tensor(x_nodes, dtype=torch.float)
            edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
            y = torch.tensor([label], dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            subject_graphs.append(data)

            aug_id = f"{subject_id}_aug{aug_idx}" if aug_idx > 0 else subject_id
            subject_metadata.append({
                "subject_id": aug_id,
                "label": label,
                "n_nodes": x.shape[0],
                "n_edges": edge_index.shape[1],
                "isolated_nodes": np.sum(np.sum(adj, axis=1) == 0),
                "density": round(nx.density(nx.from_numpy_array(adj)), 4),
                "quality_score": round(x.shape[0] * nx.density(nx.from_numpy_array(adj)), 2)
            })

        return subject_graphs, subject_metadata

    except Exception as e:
        print(f"Error with {subject_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        return None, None




print(f"Processing {len(new_files)} subjects in parallel with 8workers.")
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_subject, subject_id, fname) for subject_id, fname in new_files]
    for future in tqdm(as_completed(futures), total=len(futures)):
        g, m = future.result()
        if g is not None:
            graphs.extend(g)
            metadata.extend(m)


if graphs:
    graphs_path = os.path.join(OUTPUT_DIR, "graphs.pk_fix_short")
    if os.path.exists(graphs_path):
        with open(graphs_path, "rb") as f:
            old_graphs = pickle.load(f)
        graphs = old_graphs + graphs

    with open(graphs_path, "wb") as f:
        pickle.dump(graphs, f)

    if os.path.exists(metadata_path):
        old_metadata = pd.read_csv(metadata_path)
        metadata_df = pd.concat([old_metadata, pd.DataFrame(metadata)], ignore_index=True)
    else:
        metadata_df = pd.DataFrame(metadata)

    metadata_df.to_csv(metadata_path, index=False)
    print(f"Total of saved graphs: {len(graphs)}")

print("Graph dataset updated in data/pyg_graphs")


