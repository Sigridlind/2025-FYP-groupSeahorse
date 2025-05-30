##### USES PREPROCESSED IMAGE(hairremoval and uint8 changed, and quality check)

"""
feature_C.py

This module computes the color variation score of a lesion using its original image and corresponding binary mask.
The score is calculated as the sum of standard deviations of R, G, and B pixel values within the masked region.

Higher scores indicate more color variation — often associated with melanoma.
"""
import numpy as np
from util.inpaint_util import preprocess
from util.img_util import preprocess_mask

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

    # Apply image preprocessing: removes hair, denoises, keeps original size
    image = preprocess(image_path, apply_eq=False, apply_denoise=True, resize=False)
    
    # If preprocessing failed (e.g., bad image quality), return None
    if image is None:
        return None
    mask = preprocess_mask(mask_path)
    
    
    if mask.shape != image.shape[:2]:
        # Resize mask to match image dimensions (1-pixel differences)
        from skimage.transform import resize
        print(f"Auto-resizing mask: {mask.shape} → {image.shape[:2]}")
        mask = resize(mask, image.shape[:2], order=0, preserve_range=True, anti_aliasing=False) > 0.5

    # Apply mask to RGB image: extract only lesion pixels
    masked_pixels = image[mask]

    if masked_pixels.size == 0:
        return 0 # No lesion pixels detected

    r_std = np.std(masked_pixels[:, 0]) # std of all colours
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])
    score=round(r_std + g_std + b_std, 3) # find sum of stds
    # Normalize assuming max variation ~200 (empirical, safe upper bound)
    normalized_score = min(score/ 200.0, 1.0)
    return round(normalized_score, 3)


