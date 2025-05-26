
import numpy as np
from math import ceil, floor
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import rotate

def asymmetry_score(mask_path, degrees_step=5):
    """
    Compute the mean asymmetry score for a binary lesion mask,
    rotating in `degrees_step` increments.

    Parameters:
        mask_path (str): Path to the binary lesion mask (PNG).
        degrees_step (int): Step size for rotation in degrees (default: 5 → 72 angles).

    Returns:
        float: Mean asymmetry score across all rotated versions.
    """

    def preprocess_mask(mask): # checks if mask is binary, otherwise convert
        mask = imread(mask)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0

    def find_midpoint_v1(mask): # find midpoint in whole mask (finds based of cropped mask size)
        row_mid = mask.shape[0] / 2
        col_mid = mask.shape[1] / 2
        return row_mid, col_mid

    def find_midpoint_v4(mask): # used in crop_mask (finds based of lesion mass)
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i

    def crop_mask(mask):
        def fallback_crop(mask):
            y_nonzero, x_nonzero = np.nonzero(mask)
            if len(y_nonzero) == 0 or len(x_nonzero) == 0:
                return None
            y_min, y_max = np.min(y_nonzero), np.max(y_nonzero)
            x_min, x_max = np.min(x_nonzero), np.max(x_nonzero)
            return mask[y_min:y_max + 1, x_min:x_max + 1]

        y_nonzero, x_nonzero = np.nonzero(mask)
        if len(y_nonzero) == 0 or len(x_nonzero) == 0:
            return None

        mid = find_midpoint_v4(mask)
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
        x_dist = max(np.abs(x_lims - mid))

        x0 = max(int(mid - x_dist), 0)
        x1 = min(int(mid + x_dist), mask.shape[1])

        if x1 <= x0:
            return fallback_crop(mask)

        cropped = mask[y_lims[0]:y_lims[1] + 1, x0:x1]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return fallback_crop(mask)

        return cropped

    def cut_mask(mask): # used in get score
        col_sums = np.sum(mask, axis=0)
        row_sums = np.sum(mask, axis=1)
        active_cols = np.where(col_sums > 0)[0]
        active_rows = np.where(row_sums > 0)[0]
        if active_rows.size == 0 or active_cols.size == 0:
            return mask
        return mask[active_rows[0]:active_rows[-1]+1, active_cols[0]:active_cols[-1]+1]

    def compute_asymmetry(mask):
        row_mid, col_mid = find_midpoint_v1(mask)
        upper = mask[:ceil(row_mid), :]
        lower = mask[floor(row_mid):, :]
        left = mask[:, :ceil(col_mid)]
        right = mask[:, floor(col_mid):]
        flipped_lower = np.flip(lower, axis=0)
        flipped_right = np.flip(right, axis=1)
        hori = np.logical_xor(upper, flipped_lower)
        vert = np.logical_xor(left, flipped_right)
        total = np.sum(mask)
        if total == 0:
            return 0.0
        return round((np.sum(hori) + np.sum(vert)) / (2 * total), 4)

    def rotation_asymmetry(mask, step):
        scores = {}
        for deg in range(0, 360, step):
            rotated = rotate(mask, deg, preserve_range=True)
            binarized = rotated > 0.5
            trimmed = cut_mask(binarized)
            scores[deg] = compute_asymmetry(trimmed)
        return scores
    
    # --- Full process ---
    mask = preprocess_mask(mask_path)
    mask = crop_mask(mask)
    scores = rotation_asymmetry(mask, degrees_step)
    return round(np.mean(list(scores.values())), 4)

    


# from util.asymmetry import get_asymmetry_score

# mask_path = "lesion_masks/PAT_8_15_820_mask.png"
# score = get_asymmetry_score(mask_path)
# print("Mean asymmetry score (5° steps):", score)


