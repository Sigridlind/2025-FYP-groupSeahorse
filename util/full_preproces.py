### import cv2
### import numpy as np
### from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def preprocess(image_path, apply_eq=False, apply_denoise=False, resize=False, output_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        return None
    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect hair color and remove hair
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mask = cv2.convertScaleAbs(lap) >= np.percentile(np.abs(lap), 100)
    hair_type = "black" if np.mean(gray[mask]) < 128 else "white"
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25))
    op = cv2.MORPH_BLACKHAT if hair_type == "black" else cv2.MORPH_TOPHAT
    filtered = cv2.morphologyEx(gray, op, kernel)
    _, hmask = cv2.threshold(filtered, 10, 255, cv2.THRESH_BINARY)
    hairless = cv2.inpaint(img, hmask, 3, cv2.INPAINT_TELEA)

    # PSNR & SSIM check
    psnr = peak_signal_noise_ratio(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
    ssim = structural_similarity(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
    if psnr < 20 or ssim < 0.8:
        return None

    # Optional filters
    if apply_denoise:
        hairless = cv2.bilateralFilter(hairless, 9, 75, 75)
    if apply_eq:
        eq = cv2.equalizeHist(cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
        hairless = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    if resize:
        hairless = cv2.resize(hairless, output_size, interpolation=cv2.INTER_CUBIC)

    return hairless.astype(np.uint8)


### Example of usage (options for denoising, equalizing, resizing)
# img = preprocess("path/to/image.png", apply_eq=False, apply_denoise=False, resize=False)
