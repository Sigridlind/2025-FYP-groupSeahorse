from skimage.io import imread
from skimage.color import rgb2gray
import numpy as np
# def asymmetry_score(mask_path, degrees_step=5):
#     from skimage.io import imread
#     from skimage.color import rgb2gray
#     from skimage.transform import rotate
#     import numpy as np
#     from math import ceil, floor

#     print(f"\n[INFO] Reading mask from: {mask_path}")

#     def preprocess_mask(mask_path):
#         mask = imread(mask_path)
#         print(f"[INFO] Original mask shape: {mask.shape}")
#         if mask.ndim == 3:
#             print("[DEBUG] Converting RGB mask to grayscale.")
#             mask = rgb2gray(mask)
#         binary = mask > 0
#         print(f"[DEBUG] Binary mask created. Non-zero pixels: {np.sum(binary)}")
#         return binary

#     def find_midpoint_v1(mask):
#         row_mid = mask.shape[0] / 2
#         col_mid = mask.shape[1] / 2
#         return row_mid, col_mid

#     def find_midpoint_v4(mask):
#         summed = np.sum(mask, axis=0)
#         half_sum = np.sum(summed) / 2
#         for i, n in enumerate(np.add.accumulate(summed)):
#             if n > half_sum:
#                 return i
#         return mask.shape[1] // 2  # fallback midpoint

#     def crop_mask(mask):
#         mid = find_midpoint_v4(mask)
#         y_nonzero, x_nonzero = np.nonzero(mask)
#         if len(y_nonzero) == 0 or len(x_nonzero) == 0:
#             print("[WARN] Empty mask, cannot crop.")
#             return None
#         y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
#         x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
#         x_dist = max(np.abs(x_lims - mid))
#         x_lims = [int(mid - x_dist), int(mid + x_dist)]
#         cropped = mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]
#         print(f"[DEBUG] Cropped mask shape: {cropped.shape}")
#         return cropped

#     def cut_mask(mask):
#         col_sums = np.sum(mask, axis=0)
#         row_sums = np.sum(mask, axis=1)
#         active_cols = np.where(col_sums > 0)[0]
#         active_rows = np.where(row_sums > 0)[0]
#         if active_rows.size == 0 or active_cols.size == 0:
#             print("[WARN] Trimmed mask has no active area.")
#             return None
#         trimmed = mask[active_rows[0]:active_rows[-1]+1, active_cols[0]:active_cols[-1]+1]
#         return trimmed

#     def compute_asymmetry(mask):
#         if mask is None or mask.size == 0:
#             return None
#         row_mid, col_mid = find_midpoint_v1(mask)
#         upper = mask[:ceil(row_mid), :]
#         lower = mask[floor(row_mid):, :]
#         left = mask[:, :ceil(col_mid)]
#         right = mask[:, floor(col_mid):]
#         flipped_lower = np.flip(lower, axis=0)
#         flipped_right = np.flip(right, axis=1)
#         hori = np.logical_xor(upper, flipped_lower)
#         vert = np.logical_xor(left, flipped_right)
#         total = np.sum(mask)
#         if total == 0:
#             return None
#         score = (np.sum(hori) + np.sum(vert)) / (2 * total)
#         return round(score, 4)

#     def rotation_asymmetry(mask, step):
#         scores = {}
#         for deg in range(0, 360, step):
#             rotated = rotate(mask, deg, preserve_range=True)
#             binarized = rotated > 0.5
#             trimmed = cut_mask(binarized)
#             score = compute_asymmetry(trimmed)
#             if score is not None:
#                 scores[deg] = score
#                 print(f"[DEBUG] Rotation {deg}° → asymmetry score: {score}")
#             else:
#                 print(f"[WARN] Rotation {deg}° → invalid asymmetry score")
#         return scores

#     # --- Full process ---
#     mask = preprocess_mask(mask_path)
#     if mask is None or not np.any(mask):
#         print("[ERROR] Mask is empty or invalid.")
#         return np.nan

#     mask = crop_mask(mask)
#     if mask is None or mask.size == 0:
#         print("[ERROR] Cropped mask is empty.")
#         return np.nan

#     scores = rotation_asymmetry(mask, degrees_step)
#     if not scores:
#         print("[ERROR] All rotations failed to compute scores.")
#         return np.nan

#     final_score = round(np.mean(list(scores.values())), 4)
#     print(f"[RESULT] Final asymmetry score: {final_score}")
#     return final_score



# test_mask = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/lesion_masks/PAT_793_1512_327_mask.png"  # ← set to one that exists
# asymmetry_score(test_mask)


def crop_mask(mask):
    import numpy as np

    def fallback_crop(mask):
        """Crop using the bounding box of all non-zero pixels."""
        y_nonzero, x_nonzero = np.nonzero(mask)
        if len(y_nonzero) == 0 or len(x_nonzero) == 0:
            print("[FALLBACK] Empty mask — cannot fallback crop.")
            return None
        y_min, y_max = np.min(y_nonzero), np.max(y_nonzero)
        x_min, x_max = np.min(x_nonzero), np.max(x_nonzero)
        cropped = mask[y_min:y_max + 1, x_min:x_max + 1]
        print(f"[FALLBACK] Bounding box crop shape: {cropped.shape}")
        return cropped

    def find_midpoint_v4(mask):
        """Find midpoint based on cumulative horizontal mass."""
        summed = np.sum(mask, axis=0)
        half_sum = np.sum(summed) / 2
        for i, n in enumerate(np.add.accumulate(summed)):
            if n > half_sum:
                return i
        return mask.shape[1] // 2  # fallback

    # Original method
    y_nonzero, x_nonzero = np.nonzero(mask)
    if len(y_nonzero) == 0 or len(x_nonzero) == 0:
        print("[WARN] No lesion found in mask.")
        return None

    mid = find_midpoint_v4(mask)
    y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
    x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)])
    x_dist = max(np.abs(x_lims - mid))

    x0 = max(int(mid - x_dist), 0)
    x1 = min(int(mid + x_dist), mask.shape[1])

    if x1 <= x0:
        print(f"[WARN] Invalid symmetric crop (width 0): ({x0}, {x1}) — using fallback.")
        return fallback_crop(mask)

    cropped = mask[y_lims[0]:y_lims[1] + 1, x0:x1]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        print(f"[WARN] Cropped mask has empty dimension: {cropped.shape} — using fallback.")
        return fallback_crop(mask)

    print(f"[DEBUG] Symmetric crop shape: {cropped.shape}")
    return cropped



test_mask_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/lesion_masks/PAT_793_1512_327_mask.png"

# Read image
mask = imread(test_mask_path)
if mask.ndim == 3:
    mask = rgb2gray(mask)
mask = mask > 0  # Convert to binary mask

# Now test cropping
cropped_mask = crop_mask(mask)
