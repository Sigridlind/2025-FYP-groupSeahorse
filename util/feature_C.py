##### USES PREPROCESSED IMAGE(hairremoval and uint8 changed, and quality check)
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from full_preproces import preprocess

def color_score(image_path, mask_path):
    def preprocess_mask(mask_path): # convert to binary if not
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0

    image = preprocess(image_path, apply_eq=False, apply_denoise=False, resize=False)
    if image is None:
        return None

    mask = preprocess_mask(mask_path)
    masked_pixels = image[mask] # mask the image
    if masked_pixels.size == 0:
        return 0

    r_std = np.std(masked_pixels[:, 0]) # std of all colours
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])
    score=round(r_std + g_std + b_std, 3) # find sum of stds
    # Normalize assuming max variation ~200 (empirical, safe upper bound)
    normalized_score = min(score/ 200.0, 1.0)
    return round(normalized_score, 3)


