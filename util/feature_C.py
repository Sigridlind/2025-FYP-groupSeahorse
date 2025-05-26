##### USES PREPROCESSED IMAGE(hairremoval and uint8 changed, and quality check)
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import util.full_preproces

def color_score(image_path, mask_path):
    def preprocess_mask(mask_path):
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0

    image = util.full_preproces.preprocess(image_path, apply_eq=False, apply_denoise=True, resize=False)
    if image is None:
        return None

    mask = preprocess_mask(mask_path)
    masked_pixels = image[mask]
    if masked_pixels.size == 0:
        return 0

    r_std = np.std(masked_pixels[:, 0])
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])
    return round(r_std + g_std + b_std, 3)
