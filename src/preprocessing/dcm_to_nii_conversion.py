import os
import subprocess


def convert_dicom_to_nii(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    new_conversions = []
    total_dicom_scans = 0
    already_converted = 0

    for dirpath, _, filenames in os.walk(input_path):
        dcm_files = [f for f in filenames if f.lower().endswith('.dcm')]
        if dcm_files:
            parts = os.path.normpath(dirpath).split(os.sep)

          
            subject_id = next((p for p in parts if p.count('_') == 2), None)
            image_id = next((p for p in parts if p.startswith("I") and p[1:].isdigit()), None)

            if subject_id and image_id:
                total_dicom_scans += 1
                subject_output = os.path.join(output_path, subject_id)
                os.makedirs(subject_output, exist_ok=True)

                nii_exists = (
                    os.path.exists(subject_output) and
                    any(f.startswith(image_id) and f.endswith(".nii.gz") for f in os.listdir(subject_output))
                )

                if not nii_exists:
                    new_conversions.append((dirpath, subject_output, image_id))
                else:
                    already_converted += 1

    print(f"Dataset Info:")
    print(f"Number of DICOM scans detected: {total_dicom_scans}")
    print(f"Scans already converted into NIFTI: {already_converted}")
    print(f"Scans that still need to be converted: {len(new_conversions)}")
    if total_dicom_scans > 0:
        print(f"Percentage converted: {len(new_conversions) / total_dicom_scans * 100:.2f}%\n")
    else:
        print("No DICOM scans where detected.\n")

    for dirpath, subject_output, image_id in new_conversions:
        print(f"Converting: {dirpath} → {subject_output}")
        subprocess.run([
            "dcm2niix",
            "-z", "y",
            "-o", subject_output,
            dirpath
        ])

input_path = r"F:\00 Lucia\AML_project\ADNI" # 4TB Hard Drive with small parts of the needed scans downloaded from ADNI database
output_path = os.path.join("data", "fMRI_nii")
print(f"Output path: {output_path}")

convert_dicom_to_nii(input_path, output_path)
