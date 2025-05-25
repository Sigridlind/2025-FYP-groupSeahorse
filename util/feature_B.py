import numpy as np
from skimage import morphology
from skimage.io import imread
from skimage.color import rgb2gray
from math import pi

def border_irregularity(mask_path):
    """
    Compute the border irregularity score from a binary mask.

    Parameters:
        mask_path (str): Path to the binary mask image.

    Returns:
        float: Border irregularity score (0 = perfect circle, 1 = very irregular)
    """
    def preprocess_mask(mask): # if not binary convert to binary
        mask = imread(mask)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0

    def compute_score(mask): #compute the score for compactness 0-1. 0 being perfect circle 
        area = np.sum(mask)
        struct_el = morphology.disk(2)
        eroded = morphology.binary_erosion(mask, struct_el)
        perimeter = np.logical_xor(mask, eroded)
        perimeter_len = np.sum(perimeter)

        if perimeter_len == 0:
            return 0.0

        compactness = (4 * pi * area) / (perimeter_len ** 2) # normalized compactness score
        return round(1 - compactness, 3)

    mask = preprocess_mask(mask_path)
    return compute_score(mask)
