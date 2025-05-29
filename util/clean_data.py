import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

def clean_data(df, mask_dir, min_lesion_pixels=10, binarization_threshold=0.05):
    """
    Removes rows where:
    - the corresponding mask does not exist
    - removes imgs were segmentation masks were inaccurate
    - or the mask has fewer than `min_lesion_pixels` with intensity above `binarization_threshold`

    Parameters:
        csv_path (str): Path to your local metadata CSV
        mask_dir (str): Path to the local folder with lesion masks
        min_lesion_pixels (int): Minimum lesion pixels required to keep the row
        binarization_threshold (float): Threshold to binarize the mask (default: 0.05)

    Returns:
        None: Updates the CSV in-place
    """
    
    exclude_ids = {"PAT_488_931_321.png", "PAT_1725_3222_943.png"}
    valid_rows = []

    for idx, row in df.iterrows():
        img_id = row["img_id"]
        
        if img_id in exclude_ids:
            print(f"Excluded known problematic image: {img_id}")
            continue
        
        mask_name = img_id.replace(".png", "_mask.png")
        mask_path = os.path.normpath(os.path.join(mask_dir, mask_name))

        if not os.path.exists(mask_path):
            continue

        try:
            mask = imread(mask_path)

            if mask.ndim == 3:
                mask = rgb2gray(mask)

            binary_mask = mask > binarization_threshold
            lesion_pixels = np.sum(binary_mask)

            if lesion_pixels < min_lesion_pixels:
                print(f"Too few lesion pixels ({lesion_pixels:.0f}) in: {img_id}")
                continue

        except Exception as e:
            print(f"Error reading {img_id}: {e}")
            continue

        valid_rows.append(row)
    filtered_df = pd.DataFrame(valid_rows)
    print(f"Kept {len(filtered_df)} rows out of {len(df)}.")

    return filtered_df