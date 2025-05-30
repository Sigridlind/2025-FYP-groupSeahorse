import os
import pandas as pd
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

def clean_data(df, mask_dir, min_lesion_pixels=10, binarization_threshold=0.05):
    """
    Cleans metadata by removing rows with missing or invalid lesion masks.

    A row is removed if:
    - The corresponding mask file is missing
    - The mask has too few lesion pixels after binarization
    - The mask could not be read or processed

    Parameters:
        df (pd.DataFrame): Input metadata table with image IDs
        mask_dir (str): Directory containing segmentation mask files
        min_lesion_pixels (int): Minimum number of pixels required in the lesion mask
        binarization_threshold (float): Threshold for binarizing grayscale mask

    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid rows
    """
    # Store rows that pass all checks
    valid_rows = []

    for idx, row in df.iterrows():
        img_id = row["img_id"]
        
        mask_name = img_id.replace(".png", "_mask.png")
        mask_path = os.path.normpath(os.path.join(mask_dir, mask_name))
        
        # Skip if mask file is missing
        if not os.path.exists(mask_path):
            continue

        try:
            mask = imread(mask_path)
            
            # Convert RGB mask to grayscale if needed
            if mask.ndim == 3:
                mask = rgb2gray(mask)
            
            # Binarize mask based on threshold
            binary_mask = mask > binarization_threshold
            lesion_pixels = np.sum(binary_mask)

            if lesion_pixels < min_lesion_pixels:
                print(f"Too few lesion pixels ({lesion_pixels:.0f}) in: {img_id}")
                continue

        except Exception as e:
            print(f"Error reading {img_id}: {e}")
            continue

        valid_rows.append(row)

    # Rebuild DataFrame with only valid entries
    filtered_df = pd.DataFrame(valid_rows)
    print(f"Kept {len(filtered_df)} rows out of {len(df)}.")

    return filtered_df
