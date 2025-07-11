import os
import json
import nibabel as nib
from nilearn.image import smooth_img, clean_img, resample_to_img
from nilearn.datasets import fetch_atlas_aal
from nilearn.masking import compute_brain_mask
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random

INPUT_DIR = "data/fMRI_nii"
OUTPUT_DIR = "data/fMRI_nii_preprocessed_trial"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_DIR_plot = "outputs/subject_plots"
os.makedirs(OUTPUT_DIR_plot, exist_ok=True)


atlas = fetch_atlas_aal()
atlas_filename = atlas.maps

DEFAULT_TR = 3.0
SMOOTHING_FWHM = 0.5  
LOW_PASS = 0.1
HIGH_PASS = 0.01
STANDARDIZE = True

def get_tr_from_json(nii_path):
    json_path = nii_path.replace(".nii.gz", ".json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                tr = metadata.get("RepetitionTime")
                if tr:
                    return float(tr)
        except Exception as e:
            print(f"JSON couldn't be read {json_path}: {e}")
    return DEFAULT_TR


def preprocess_single_file(input_path, output_path):
    try:
        tr = get_tr_from_json(input_path)
        print(f"Processing: {os.path.basename(input_path)}, TR = {tr}")

        img = nib.load(input_path)
        data = img.get_fdata()
        print(f"Original: shape={data.shape}, min={np.min(data):.3f}, max={np.max(data):.3f}, mean={np.mean(data):.3f}")
        print(f"NaNs: {np.isnan(data).sum()}, Zeros: {(data == 0).sum()}")

        brain_mask = compute_brain_mask(img)
        mask_data = brain_mask.get_fdata().astype(bool)
        brain_voxels = data[mask_data]
        print(f"Brain voxels: {brain_voxels.size}, Mean: {np.mean(brain_voxels):.3f}, Zeros: {(brain_voxels == 0).sum()}")

        cleaned = clean_img(
            img,
            t_r=tr,
            detrend=True,
            standardize=True, 
            low_pass=LOW_PASS,
            high_pass=HIGH_PASS,
        )
        cleaned_data = cleaned.get_fdata()
        print(f"Cleaned: shape={cleaned_data.shape}, min={np.min(cleaned_data):.3f}, max={np.max(cleaned_data):.3f}, mean={np.mean(cleaned_data):.3f}")
        print(f"NaNs: {np.isnan(cleaned_data).sum()}, Zeros: {(cleaned_data == 0).sum()}")

        smoothed = smooth_img(cleaned, fwhm=SMOOTHING_FWHM)
        smoothed_data = smoothed.get_fdata()
        print(f"Smoothed: shape={smoothed_data.shape}, min={np.min(smoothed_data):.3f}, max={np.max(smoothed_data):.3f}, mean={np.mean(smoothed_data):.3f}")
        print(f" NaNs: {np.isnan(smoothed_data).sum()}, Zeros: {(smoothed_data == 0).sum()}")

        resampled_atlas = resample_to_img(atlas_filename, smoothed, interpolation='nearest')
        resampled_data = resampled_atlas.get_fdata()
        print(f"Resampled Atlas: shape={resampled_data.shape}, min={np.min(resampled_data):.3f}, max={np.max(resampled_data):.3f}, mean={np.mean(resampled_data):.3f}")
        print(f"NaNs: {np.isnan(resampled_data).sum()}, Zeros: {(resampled_data == 0).sum()}")


        nib.save(smoothed, output_path)

        brain_ts = cleaned_data[mask_data]  
        brain_var = np.var(brain_ts, axis=0)
        var_mean = np.mean(brain_var)
        var_std = np.std(brain_var)
        var_min = np.min(brain_var)
        var_max = np.max(brain_var)
        print(f"Checking quality: Brain variance after cleaning: mean={var_mean:.6f}, std={var_std:.6f}, min={var_min:.6f}, max={var_max:.6f}")
        if var_mean < 1e-4 or var_max < 1e-3:
            print("WARNING! Signal could be dead: variability is to low after clean_img")


        near_constant_voxels = np.sum(brain_var < 1e-6)
        print(f"Checking quality: Nearly constant voxels: {near_constant_voxels}/{brain_ts.shape[0]} ({near_constant_voxels / brain_ts.shape[0] * 100:.2f}%)")
        if near_constant_voxels > 0.5 * brain_ts.shape[0]:
            print("WARNING! More than 50% of brain voxels have a flat signal!")

        unique_labels = np.unique(resampled_data)
        missing_rois = set(np.unique(nib.load(atlas_filename).get_fdata())) - set(unique_labels)
        print(f"Checking quality: ROIs kept after resampling: {len(unique_labels)} / {len(set(np.unique(nib.load(atlas_filename).get_fdata())))}")
        if len(missing_rois) > 10:
            print(f"WARNING! Too many ROIs were lost after resampling!: {sorted(missing_rois)}")

        nan_ratio = np.isnan(cleaned_data).sum() / cleaned_data.size
        if nan_ratio > 0:
            print(f"Checking quality: NaNs detected after cleaning: {nan_ratio * 100:.2f}% of the data")
        
        # PLOT: BOLD Signal in a random voxel from the brain to ensure that the signal is not being killed during preprocessing
        if brain_ts.ndim == 2 and brain_ts.shape[1] > 0:
            rand_voxel_idx = random.randint(0, brain_ts.shape[0] - 1)
            plt.plot(brain_ts[rand_voxel_idx, :])
            plt.title("BOLD Signal in a random voxel after cleaning")
            plt.xlabel("Time")
            plt.ylabel("Intensity")
            plt.grid(True)
            plt.tight_layout()
            debug_plot_path = os.path.join(OUTPUT_DIR_plot, f"bold_plot_{os.path.basename(input_path).replace('.nii.gz', '')}.png")
            plt.savefig(debug_plot_path)
            plt.close()
            print(f"Plot saved: {debug_plot_path}")
        
        print(f"fMRI saved in: {output_path}")


    except Exception as e:
        print(f"Error with {input_path}: {e}")


input_paths = []
output_paths = []

total_nii_files = 0
already_preprocessed = 0
pending_details = []

for root, _, files in os.walk(INPUT_DIR):
    for file in files:
        if file.endswith(".nii.gz"):
            total_nii_files += 1
            input_path = os.path.join(root, file)
            base_name = os.path.splitext(os.path.splitext(file)[0])[0]
            processed_name = base_name + "_processed.nii.gz"
            output_path = os.path.join(OUTPUT_DIR, processed_name)

            if not os.path.exists(output_path):
                input_paths.append(input_path)
                output_paths.append(output_path)

                parts = os.path.normpath(input_path).split(os.sep)
                subject_id = next((p for p in parts if p.count('_') == 2), "UNKNOWN_SUBJECT")
                image_id = next((p for p in parts if p.startswith("I") and p[1:].isdigit()), base_name.split('_')[0])
                pending_details.append((subject_id, image_id, input_path, output_path))
            else:
                already_preprocessed += 1

print(f"Total number of detected .nii.gz files: {total_nii_files}")
print(f"Already preprocessed: {already_preprocessed}")
print(f"Pending to preprocess: {len(input_paths)}")
if total_nii_files > 0:
    print(f"Pending percentage: {len(input_paths) / total_nii_files * 100:.2f}%\n")

if pending_details:
    print("Pending files to preprocess:\n")
    for subj, img, in_path, out_path in pending_details:
        print(f"Subject: {subj}")
        print(f"ImageID: {img}")
        print(f"NIfTI Path: {in_path}")
        print(f"Expected Output: {out_path}\n")
else:
    print("No new files to preprocess.\n")

if input_paths:
    Parallel(n_jobs=-1)(
        delayed(preprocess_single_file)(in_path, out_path)
        for in_path, out_path in zip(input_paths, output_paths)
    )
    print("Preprocessing completed in parallel.")
