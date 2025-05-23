import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray

def preprocess_mask(mask_path):
    #Reads binary maskfile
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = rgb2gray(mask)
    return mask > 0

def read_rgb_image(image_path):
    #Reads original RGB image (Not greyscale))
    image = imread(image_path)
    if image.dtype != np.uint8:
        image = (255 * image).astype(np.uint8)
    return image

def color_score(image, mask):
    #Computes color standardvariation from R, G, B chanels in the mask
    #Returns sum of the standardsdeviations as color_score
    masked_pixels = image[mask]
    if masked_pixels.size == 0:
        return 0

    r_std = np.std(masked_pixels[:, 0])
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])

    score = round(r_std + g_std + b_std, 3)
    return score

if __name__ == "__main__":
    img_path = "data/PAT_8_15_820.png"
    mask_path = "lesion_masks/PAT_8_15_820_mask.png"

    image = read_rgb_image(img_path)
    mask = preprocess_mask(mask_path)

    print("Color score:", color_score(image, mask))
