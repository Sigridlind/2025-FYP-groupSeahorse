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

#image = read_rgb_image(img_path)
#mask = preprocess_mask(mask_path)
#print("Color score:", color_score(image, mask))




import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from full_preprocess import preprocess  # Import your preprocessing function

def preprocess_mask(mask_path):
    mask = imread(mask_path)
    if mask.ndim == 3:
        mask = rgb2gray(mask)
    return mask > 0

def color_score(image, mask):
    masked_pixels = image[mask]
    if masked_pixels.size == 0:
        return 0
    r_std = np.std(masked_pixels[:, 0])
    g_std = np.std(masked_pixels[:, 1])
    b_std = np.std(masked_pixels[:, 2])
    return round(r_std + g_std + b_std, 3)

# Example usage
image_path = "path/to/image.png"
mask_path = "path/to/image_mask.png"

image = preprocess(image_path, apply_eq=False, apply_denoise=True, resize=False)
if image is not None:
    mask = preprocess_mask(mask_path)
    print("Color score:", color_score(image, mask))






