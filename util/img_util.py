import cv2
from skimage.io import imread
from skimage.color import rgb2gray

def readImageFile(file_path):
    """
    Reading an image and returning the RGB, grayscale, and original BGR versions.
    """
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

def preprocess_mask(mask_path):  
    # The path to the binary or grayscale mask image
        """
        Loading a mask and converting it to binary if it's not. Converting to grayscale if RGB.
        """
        # Convert RGB mask to grayscale if needed
        mask = imread(mask_path)
        if mask.ndim == 3:
            mask = rgb2gray(mask)
        return mask > 0
