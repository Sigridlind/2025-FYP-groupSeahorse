import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
import util.feature_C
import util.full_preproces

import cv2



# ----------- CONFIGURE TEST FILES -----------
image_path = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/EDA/imgs/PAT_1549_1882_230.png"
mask_path  = "C:/Users/Lenovo/OneDrive - ITU/Uni/2. Semester/Projects DS/Project/lesion_masks/PAT_1549_1882_230_mask.png"

# ----------- HELPER: Raw test without pipeline -----------

raw_image = cv2.imread(image_path)
if raw_image is None:
    print("[DEBUG] cv2.imread failed. Image file not found or unreadable.")
else:
    print(f"[DEBUG] cv2.imread success. Shape: {raw_image.shape}")
    
    
def test_color_score(image_path, mask_path):
    def preprocess_mask(mask_path):
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0

    print("[TEST] Loading image...")
    image = preprocess(image_path, apply_eq=False, apply_denoise=True, resize=False)
    if image is None:
        print("[ERROR] Could not load or preprocess image.")
        return

    print(f"[INFO] Preprocessed image shape: {image.shape}")
    mask = preprocess_mask(mask_path)
    print(f"[INFO] Loaded mask shape: {mask.shape}")

    if image.shape[:2] != mask.shape:
        print("[ERROR] Image and mask dimensions do not match.")
        return

    masked_pixels = image[mask]
    print(f"[INFO] Number of masked pixels: {masked_pixels.shape[0]}")

    if masked_pixels.size == 0:
        print("[WARNING] Masked region is empty. Returning 0.")
        print("Color score: 0.000")
        return

    try:
        r_std = np.std(masked_pixels[:, 0])
        g_std = np.std(masked_pixels[:, 1])
        b_std = np.std(masked_pixels[:, 2])
        score = round(r_std + g_std + b_std, 3)
        print(f"[RESULT] Color score: {score}")
    except IndexError:
        print("[ERROR] Masked pixels are not 3-channel RGB. Shape:", masked_pixels.shape)

def preprocess(image_path, apply_eq=False, apply_denoise=True, resize=False):
    import cv2

    print(f"[preprocess] Attempting to read image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print("[preprocess] cv2.imread failed.")
        return None
    print(f"[preprocess] Image loaded, shape: {image.shape}, dtype: {image.dtype}")

    # Example: if you convert to uint8 or RGB here
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("[preprocess] Not a 3-channel RGB image. Aborting.")
        return None

    # Hair removal or preprocessing steps
    # Add logging around each step:
    # e.g.
    # image = removeHair(image)
    # print("[preprocess] Hair removed")

    # At the end, just before return:
    return image  # just test r



# ----------- Run the test -----------
if __name__ == "__main__":
    test_color_score(image_path, mask_path)
