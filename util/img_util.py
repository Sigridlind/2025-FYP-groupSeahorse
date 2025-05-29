import random

import cv2


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


def saveImageFile(img_rgb, file_path):
    try:
        # convert BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save the image
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False


class ImageDataLoader:
    def __init__(self, directory, shuffle=False, transform=None):
        self.directory = directory
        self.shuffle = shuffle
        self.transform = transform

        # get a sorted list of all files in the directory
        # fill in with your own code below

        if not self.file_list:
            raise ValueError("No image files found in the directory.")

        # shuffle file list if required
        if self.shuffle:
            random.shuffle(self.file_list)

        # get the total number of batches
        self.num_batches = len(self.file_list)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        # fill in with your own code below
        pass
