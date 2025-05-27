"""
feature_B.py

This module computes the border irregularity score of a binary lesion mask using a compactness formula.
The more irregular the lesion border, the lower the compactness — and the higher the returned score (0 to 1).

A perfect circle returns a score near 0. More complex, jagged lesions return scores closer to 1.
"""

import numpy as np
from skimage import morphology
from skimage.io import imread
from skimage.color import rgb2gray
from math import pi

def border_irregularity(mask_path):
    """
    Computes the border irregularity score for a binary lesion mask.

    Parameters:
        mask_path (str): Path to the binary lesion mask (.png).

    Returns:
        float: Irregularity score in the range [0, 1],
               where 0 indicates a perfectly circular (regular) shape,
               and values closer to 1 indicate more irregular borders.
    """
    def preprocess_mask(mask):
        """
        Loads the mask and converts it to binary (True/False).
        If RGB, it's converted to grayscale first.
        """
        mask = imread(mask)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0

    def compute_score(mask):
        """
        Computes the compactness-based irregularity score using:

            score = 1 - (4π * area) / (perimeter²)

        A perfect circle has a compactness of 1 → score = 0.
        More irregular shapes have lower compactness → higher score.
        """
        # Total number of lesion pixels
        area = np.sum(mask)
        
        # Detect the lesion border by removing inner pixels (eroded mask) and comparing to the original mask
        struct_el = morphology.disk(2)
        eroded = morphology.binary_erosion(mask, struct_el)
        perimeter = np.logical_xor(mask, eroded)
        perimeter_len = np.sum(perimeter)

        if perimeter_len == 0:
            return 0.0 # Avoid division by zero

        compactness = (4 * pi * area) / (perimeter_len ** 2) # normalized compactness score
        return round(1 - compactness, 3)

    mask = preprocess_mask(mask_path)
    return compute_score(mask)
