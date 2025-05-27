
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
        row_mid = mask.shape[0] / 2 # half in y-axis
        col_mid = mask.shape[1] / 2 # half in x-axis
        return row_mid, col_mid

    def find_midpoint_v4(mask): # used in crop_mask (finds half x-axis based of lesion mass)
        summed = np.sum(mask, axis=0) # sum of mask
        half_sum = np.sum(summed) / 2 # divide it
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i # halfpoint x-axis

  def crop_mask(mask): # new
        mid = find_midpoint_v4(mask)
        y_nonzero, x_nonzero = np.nonzero(mask) # find 
        y_min, y_max = np.min(y_nonzero), np.max(y_nonzero) # find limits
        x_min, x_max = np.min(x_nonzero), np.max(x_nonzero) # find limits

        x_dist = max(abs(x_min - mid), abs(x_max - mid)) # find max of distance between min/max and mid

        x0 = max(int(mid - x_dist), 0) # find max of differrence between mid and x distance
        x1 = min(int(mid + x_dist), mask.shape[1]) # find min of difference
        
        if x1 <= x0: # if min greater than max difference
            # fallback to tight bounding box crop
            return mask[y_min:y_max+1, x_min:x_max+1]

        cropped = mask[y_min:y_max+1, x0:x1]
        return cropped

    def cut_mask(mask): #  |
        col_sums = np.sum(mask, axis=0) # sum of columns (0=black, 1=white) so only the lesion masks
        row_sums = np.sum(mask, axis=1) # sum of row, only the lesions mask
        active_cols = np.where(col_sums > 0)[0] # finds indexes of lesion pixels(at least one) in columns 
        active_rows = np.where(row_sums > 0)[0] # same but in rows
        if active_rows.size == 0 or active_cols.size == 0:
            return mask # if no active
        return mask[active_rows[0]:active_rows[-1]+1, active_cols[0]:active_cols[-1]+1] # mask by active coloums

    def compute_asymmetry(mask):
        row_mid, col_mid = find_midpoint_v1(mask) # find geometric midpoint
        upper = mask[:ceil(row_mid), :] # upper half of cropped mask
        lower = mask[floor(row_mid):, :] # lower half
        left = mask[:, :ceil(col_mid)] # left side
        right = mask[:, floor(col_mid):] # right side
        flipped_lower = np.flip(lower, axis=0) # flip it
        flipped_right = np.flip(right, axis=1) # flip it
        hori = np.logical_xor(upper, flipped_lower) # uses XOR to compare halves upper and lower
        vert = np.logical_xor(left, flipped_right) # same but left and right
        total = np.sum(mask) # sum of mask
        if total == 0:
            return 0.0
        return round((np.sum(hori) + np.sum(vert)) / (2 * total), 4) # Sum of differences and normalized

    def rotation_asymmetry(mask, step): # rotate mask 5 degrees
        scores = {}
        for deg in range(0, 360, step):
            rotated = rotate(mask, deg, preserve_range=True) # rotate
            binarized = rotated > 0.5 # convert every pixel that is not completely black or white to black or white
            trimmed = cut_mask(binarized) # cut mask (crop)
            scores[deg] = compute_asymmetry(trimmed)
        return scores # return scores of all rotated assymetrical scores
    
    # --- Full process ---
    mask = preprocess_mask(mask_path)
    mask = crop_mask(mask)
    scores = rotation_asymmetry(mask, degrees_step)
    return round(np.mean(list(scores.values())), 4) # return mean assymetri of mask

    
