import numpy as np
from skimage import morphology
from math import pi
from skimage.io import imread
from skimage.color import rgb2gray

def preprocess_image(image_path):
    mask = imread(image_path)
    if mask.ndim == 3:
        mask = rgb2gray(mask)
    mask = mask > 0  
    return mask

def get_compactness(mask):
    area = np.sum(mask)
    struct_el = morphology.disk(3)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.sum(np.logical_xor(mask, mask_eroded))  # Fix here

    if area == 0:
        return 0

    compactness = (perimeter ** 2) / (4 * pi * area)
    return compactness

def compactness_score(mask):
    A = np.sum(mask)
    struct_el = morphology.disk(2)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    perimeter = np.logical_xor(mask, mask_eroded)  
    l = np.sum(perimeter)

    if l == 0:
        return 0

    compactness = (4 * pi * A) / (l ** 2)
    score = round(1 - compactness, 3)
    return score

image_path = "/lesion_masks/PAT_8_15_820_mask.png" 
mask = preprocess_image(image_path)
print("Compactness:", get_compactness(mask))
print("Score:", compactness_score(mask))
