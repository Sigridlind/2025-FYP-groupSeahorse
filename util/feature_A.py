# asymmetry score 

import numpy as np
from math import ceil, floor
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rotate
from tkinter import messagebox, Tk

def preprocess_image(image_path):
    mask = imread(image_path)
    if mask.ndim == 3:
        mask = rgb2gray(mask)
    mask = mask > 0  
    return mask

def find_midpoint_v4(mask):
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i

def crop(mask):
        mid = find_midpoint_v4(mask)
        y_nonzero, x_nonzero = np.nonzero(mask)
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        x_dist = max(np.abs(x_lims - mid))
        x_lims = [mid - x_dist, mid+x_dist]
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def find_midpoint_v1(mask):
    row_mid = mask.shape[0] / 2
    col_mid = mask.shape[1] / 2
    return row_mid, col_mid


def cut_mask(mask): # binarize rotated mask
    col_sums = np.sum(mask, axis=0)
    row_sums = np.sum(mask, axis=1)

    active_cols = []
    for index, col_sum in enumerate(col_sums):
        if col_sum != 0:
            active_cols.append(index)

    active_rows = []
    for index, row_sum in enumerate(row_sums):
        if row_sum != 0:
            active_rows.append(index)

    col_min = active_cols[0]
    col_max = active_cols[-1]
    row_min = active_rows[0]
    row_max = active_rows[-1]

    cut_mask_ = mask[row_min:row_max+1, col_min:col_max+1]

    return cut_mask_ 

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


