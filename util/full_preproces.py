import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_hair_color(gray_img):
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    abs_lap = cv2.convertScaleAbs(lap)
    threshold_val = np.percentile(abs_lap, 100)
    high_response_mask = abs_lap >= threshold_val
    hair_pixels = gray_img[high_response_mask]
    mean_intensity = np.mean(hair_pixels)
    return "black" if mean_intensity < 128 else "white"

def removeHair_auto(gray_img, original_img, kernel_size=25, threshold=10, radius=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    hair_type = detect_hair_color(gray_img)
    if hair_type == "black":
        filtered = cv2.morphologyEx(gray_img, cv2.MORPH_BLACKHAT, kernel)
    else:
        filtered = cv2.morphologyEx(gray_img, cv2.MORPH_TOPHAT, kernel)
    _, hair_mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    img_hairless = cv2.inpaint(original_img, hair_mask, radius, cv2.INPAINT_TELEA)
    return hair_type, filtered, hair_mask, img_hairless

def cut_im_by_mask(image, mask, padding=5):
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("Mask is empty, cannot crop.")
    x, y, w, h = cv2.boundingRect(coords)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)
    return image[y:y+h, x:x+w]

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def evaluate_quality(original, processed, mask=None):
    # Convert to grayscale for SSIM
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # If mask is provided, apply it
    if mask is not None:
        original_gray = cv2.bitwise_and(original_gray, original_gray, mask=mask)
        processed_gray = cv2.bitwise_and(processed_gray, processed_gray, mask=mask)

    # SSIM
    ssim_score = structural_similarity(original_gray, processed_gray)
    # PSNR
    psnr_score = peak_signal_noise_ratio(original_gray, processed_gray)

    return psnr_score, ssim_score




def preprocess(image_path, mask_path, apply_eq=False, apply_denoise=False, show=True):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise ValueError("Image or mask could not be loaded.")

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hair_type, filtered, hair_mask, img_hairless = removeHair_auto(gray, img)
    psnr, ssim = evaluate_quality(original, img_hairless, mask)
    if psnr<30 and ssim<0.8:
        print(f"PSNR:{psnr} and SSIM:{ssim}, not good quality")
        return 

    if apply_denoise:
        img_hairless = cv2.bilateralFilter(img_hairless, d=9, sigmaColor=75, sigmaSpace=75)

    if apply_eq:
        gray_hairless = cv2.cvtColor(img_hairless, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray_hairless)
        img_hairless = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    masked_img = cv2.bitwise_and(img_hairless, img_hairless, mask=mask)
    cropped_img = cut_im_by_mask(masked_img, mask)

    if show:
        plt.figure(figsize=(24, 5))
        titles = [
            "Original",
            f"{hair_type.title()}-hat Filter",
            "Hair Mask",
            "Hair-Removed Full Image",
            "Cropped Lesion"
        ]
        images = [
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            filtered,
            hair_mask,
            cv2.cvtColor(img_hairless, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
        ]
        for i, (img_disp, title) in enumerate(zip(images, titles)):
            plt.subplot(1, 5, i + 1)
            if len(img_disp.shape) == 2:
                plt.imshow(img_disp, cmap='gray')
            else:
                plt.imshow(img_disp)
            plt.title(title)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    print("PSNR: ", psnr)
    print("SSIM: ", ssim)
    return cropped_img

preprocess(
    image_path='/Users/joakimandersen/Desktop/Projects in DS/imgs_part_1/PAT_8_15_820.png',
    mask_path='/Users/joakimandersen/Desktop/Projects in DS/lesion_masks/PAT_8_15_820_mask.png',
    apply_eq=False,
    apply_denoise=True,
    show=True
)
