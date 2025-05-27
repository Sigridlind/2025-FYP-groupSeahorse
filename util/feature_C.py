##### USES PREPROCESSED IMAGE(hairremoval and uint8 changed, and quality check)

"""
feature_C.py

This module computes the color variation score of a lesion using its original image and corresponding binary mask.
The score is calculated as the sum of standard deviations of R, G, and B pixel values within the masked region.

Higher scores indicate more color variation — often associated with melanoma.
"""
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import util.full_preproces

def color_score(image_path, mask_path):
    """
    Computes a color variation score based on RGB channel standard deviations
    inside the masked lesion area of a preprocessed image.

    Parameters:
        image_path (str): Path to the original lesion image.
        mask_path (str): Path to the corresponding binary lesion mask.

    Returns:
        float: Sum of RGB standard deviations inside the lesion (rounded to 3 decimals).
               Returns 0 if the mask is empty or None if preprocessing fails.
    """
    def preprocess_mask(mask_path):
        """
        Loads the mask and binarizes it (True for lesion pixels).
        Converts RGB to grayscale if necessary.
        """
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0
    
    # Apply image preprocessing: removes hair, denoises, keeps original size
    image = util.full_preproces.preprocess(image_path, apply_eq=False, apply_denoise=True, resize=False)
    
    # If preprocessing failed (e.g., bad image quality), return None
    if image is None:
        return None
    mask = preprocess_mask(mask_path)
    
    # Apply mask to RGB image: extract only lesion pixels
    masked_pixels = image[mask]

    if masked_pixels.size == 0:
        return 0 # No lesion pixels detected

    r_std = np.std(masked_pixels[:, 0])
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])
    return round(r_std + g_std + b_std, 3)
