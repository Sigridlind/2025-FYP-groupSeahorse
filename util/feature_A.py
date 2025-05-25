# asymmetry score 

import numpy as np
from math import ceil, floor
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rotate

def preprocess_image(image_path):
    mask = imread(image_path)
    if mask.ndim == 3:
        mask = rgb2gray(mask)
    mask = mask > 0  
    return mask

def find_midpoint_v1(image):
    row_mid = image.shape[0] / 2
    col_mid = image.shape[1] / 2
    return row_mid, col_mid

def cut_mask(mask):
    return mask  

def asymmetry(mask):
    row_mid, col_mid = find_midpoint_v1(mask)

    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)
    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n):
    asymmetry_scores = {}
    for i in range(n):
        degrees = 90 * i / n
        rotated_mask = rotate(mask, degrees)
        cutted_mask = cut_mask(rotated_mask)
        asymmetry_scores[degrees] = asymmetry(cutted_mask)
    return asymmetry_scores

def mean_asymmetry(mask, rotations=30):
    scores = rotation_asymmetry(mask, rotations)
    return round(sum(scores.values()) / len(scores), 4)

def best_asymmetry(mask, rotations=30):
    scores = rotation_asymmetry(mask, rotations)
    return round(min(scores.values()), 4)

def worst_asymmetry(mask, rotations=30):
    scores = rotation_asymmetry(mask, rotations)
    return round(max(scores.values()), 4)

# Change path
image_path = "/lesion_masks\PAT_8_15_820_mask.png"
mask = preprocess_image(image_path)
score = asymmetry(mask)

print("Asymmetry score:", score)
print("Mean asymmetry:", mean_asymmetry(mask))
print("Best asymmetry:", best_asymmetry(mask))
print("Worst asymmetry:", worst_asymmetry(mask))

