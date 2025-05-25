import cv2
import numpy as np
import os
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# --------------------------------------------------------
# Purpose:
# Preprocess a folder of dermoscopic skin lesion images by:
# - Removing hair
# - Applying binary lesion masks
# - Cropping around the lesion
# - Optional denoising, histogram equalization, CLAHE, and resizing
# - Quality-checking based on PSNR and SSIM
# - Saving clean lesion crops to a new folder
# --------------------------------------------------------


# --- Hair Removal ---
def detect_hair_color(gray_img):
    lap = cv2.Laplacian(gray_img, cv2.CV_64F)
    abs_lap = cv2.convertScaleAbs(lap)
    high_response_mask = abs_lap >= np.percentile(abs_lap, 100)
    hair_pixels = gray_img[high_response_mask]
    mean_intensity = np.mean(hair_pixels)
    return "black" if mean_intensity < 128 else "white"

def removeHair_auto(gray_img, original_img, kernel_size=25, threshold=10, radius=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    hair_type = detect_hair_color(gray_img)
    filtered = cv2.morphologyEx(gray_img,
                                 cv2.MORPH_BLACKHAT if hair_type == "black" else cv2.MORPH_TOPHAT,
                                 kernel)
    _, hair_mask = cv2.threshold(filtered, threshold, 255, cv2.THRESH_BINARY)
    img_hairless = cv2.inpaint(original_img, hair_mask, radius, cv2.INPAINT_TELEA)
    return img_hairless

# --- Cropping ---
def crop_around_mask(image, mask, padding=5):
    coords = cv2.findNonZero(mask)
    if coords is None:
        raise ValueError("Mask is empty; cannot crop.")
    x, y, w, h = cv2.boundingRect(coords)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)
    return image[y:y+h, x:x+w]

# --- Quality Evaluation ---
def evaluate_quality(original, processed, mask=None):
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        if mask.shape != original_gray.shape:
            mask = cv2.resize(mask, (original_gray.shape[1], original_gray.shape[0]))
        original_gray = cv2.bitwise_and(original_gray, original_gray, mask=mask)
        processed_gray = cv2.bitwise_and(processed_gray, processed_gray, mask=mask)

    ssim = structural_similarity(original_gray, processed_gray)
    psnr = peak_signal_noise_ratio(original_gray, processed_gray)
    return psnr, ssim

# --- Main Pipeline ---
def process_images_with_quality_check(
    image_dir,
    mask_dir,
    output_dir,
    resize_output=True,
    output_size=(224, 224),
    apply_denoise=True,
    apply_equalization=False,
    padding=5,
    psnr_thresh=20,
    ssim_thresh=0.8
):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(image_dir):
        if not fname.lower().endswith(('.png', '.jpg')):
            continue

        base_name = os.path.splitext(fname)[0]
        mask_name = base_name + "_mask.png"
        image_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            print(f"Skipping {fname} — no mask.")
            continue

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"Skipping {fname} — failed to load.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hairless = removeHair_auto(gray, image)

        # Quality check
        psnr, ssim = evaluate_quality(image, hairless, mask)
        if psnr < psnr_thresh or ssim < ssim_thresh:
            print(f"❌ Skipping(low_quality) {fname}: PSNR={psnr:.2f}, SSIM={ssim:.3f}")
            continue

        # Optional filters
        if apply_denoise:
            hairless = cv2.bilateralFilter(hairless, d=9, sigmaColor=75, sigmaSpace=75)
        if apply_equalization:
            eq = cv2.equalizeHist(cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
            hairless = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

        # Mask + Crop
        masked = cv2.bitwise_and(hairless, hairless, mask=mask)
        try:
            cropped = crop_around_mask(masked, mask, padding=padding)
        except ValueError:
            print(f"Skipping {fname} — empty mask.")
            continue

        # Resize
        if resize_output:
            cropped = cv2.resize(cropped, output_size, interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(output_dir, base_name + "_final.png")
        cv2.imwrite(save_path, cropped)
        print(f"✅ Saved: {save_path} | PSNR={psnr:.2f} | SSIM={ssim:.3f}")

# --- Run ---
process_images_with_quality_check(
    image_dir='/Users/joakimandersen/Desktop/Projects in DS/2025-FYP-groupSeahorse/data',
    mask_dir="/Users/joakimandersen/Desktop/Projects in DS/lesion_masks",
    output_dir="/Users/joakimandersen/Desktop/Projects in DS/final_processed",
    resize_output=True,
    output_size=(224, 224),
    apply_denoise=False,
    apply_equalization=False,
    padding=10,
    psnr_thresh=20,
    ssim_thresh=0.8
)
