import cv2
from skimage.io import imread
from skimage.color import rgb2gray

def readImageFile(file_path):
    # read image as an 8-bit array
    bgr = cv2.imread(file_path)
    
    if bgr is None: # if no image return none
        return None
    
    original = bgr.copy() # copy, for comparing later

    # convert to RGB
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img, gray, original

def preprocess_mask(mask_path): # convert to binary if not
        """
        Loads the mask and binarizes it (True for lesion pixels).
        Converts RGB to grayscale if necessary.
        """
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0
