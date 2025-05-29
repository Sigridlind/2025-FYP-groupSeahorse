"""
feature_B.py

This module computes the border irregularity score of a binary lesion mask using a compactness formula.
The more irregular the lesion border, the lower the compactness — and the higher the returned score (0 to 1).

A perfect circle returns a score near 0. More complex, jagged lesions return scores closer to 1.
"""

import numpy as np
from skimage import morphology
from math import pi
from util.img_util import preprocess_mask

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
    
    def compute_score(mask):
        """
        Computes the compactness-based irregularity score using:

            score = 1 - (4π * area) / (perimeter²)

        A perfect circle has a compactness of 1 → score = 0.
        More irregular shapes have lower compactness → higher score.
        """
        
        area = np.sum(mask) # Total number of lesion pixels
        
        # Detect the lesion border by removing inner pixels (eroded mask) and comparing to the original mask
        struct_el = morphology.disk(2) # structuring element (cross)
        eroded = morphology.binary_erosion(mask, struct_el) # shrinks the lesion a little
        perimeter = np.logical_xor(mask, eroded) # finds perimeter by XOR
        perimeter_len = np.sum(perimeter) # length of perimeter

        if perimeter_len == 0:
            return 0.0 # Avoid division by zero

        compactness = (4 * pi * area) / (perimeter_len ** 2) # normalized compactness score
        return round(1 - compactness, 3)

    mask = preprocess_mask(mask_path)
    return compute_score(mask)
