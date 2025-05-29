"""
full_preprocess.py

This module defines a preprocessing pipeline for dermoscopic images.
It removes hair artifacts using morphological filtering and inpainting,
optionally applies denoising and histogram equalization, and resizes the image if requested.

Used as part of feature extraction, especially for computing color features inside lesion regions.
"""

### hair removal + check quality after removal. Optional denoise, equalizing and resizing.
def preprocess(image_path, apply_eq=False, apply_denoise=False, resize=False, output_size=(224, 224)):
    """
    Preprocesses a lesion image by removing hair, optionally denoising, equalizing, and resizing it.

    Parameters:
        image_path (str): Path to the input RGB image.
        apply_eq (bool): If True, apply histogram equalization (grayscale only).
        apply_denoise (bool): If True, apply bilateral filtering to denoise the image.
        resize (bool): If True, resize the image to a fixed size.
        output_size (tuple): Target size for resizing (default: 224x224).

    Returns:
        np.ndarray or None: Preprocessed RGB image, or None if quality check fails.
    """
    import cv2
    import numpy as np
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from img_util import readImageFile
    
    # Reads images as RGB, gray, and an original
    img, gray, original = readImageFile(image_path)
    

    # Detect hair color, remove hair and applying morphological filter (Blackhat, Tophat)
    lap = cv2.Laplacian(gray, cv2.CV_64F) # laplacian filter, edge detection(rapid intensity changes)
    mask = cv2.convertScaleAbs(lap) >= np.percentile(np.abs(lap), 98) # create binary mask of image (hairpixels) preferably
    
    edge_values = gray[mask]
    # deteremines if dark or light hair based of mean intensity in edges
    if edge_values.size == 0:
        hair_type = "black"
    else:
        if np.mean(gray[mask]) < 128:
            hair_type = "black" 
        else: 
            hair_type = "white"
    
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (25, 25)) # morphological structuring element (cross) 25x25
    op = cv2.MORPH_BLACKHAT if hair_type == "black" else cv2.MORPH_TOPHAT # 
    filtered = cv2.morphologyEx(gray, op, kernel) # filters the hairs detected by blackhat or tophat(whitehat)
    _, hmask = cv2.threshold(filtered, 10, 255, cv2.THRESH_BINARY) # converts to binary hair mask
    hairless = cv2.inpaint(img, hmask, 3, cv2.INPAINT_TELEA) # inpain the hairmasks
    

    # PSNR & SSIM check 
    psnr = peak_signal_noise_ratio(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
    ssim = structural_similarity(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
    if psnr < 15 or ssim < 0.5:
        return None

    # Optional filters
    if apply_denoise: # bilateral
        hairless = cv2.bilateralFilter(hairless, 9, 75, 75)
    if apply_eq: # histogram equalization
        eq = cv2.equalizeHist(cv2.cvtColor(hairless, cv2.COLOR_BGR2GRAY))
        hairless = cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)
    if resize: # resize to this 
        hairless = cv2.resize(hairless, output_size, interpolation=cv2.INTER_CUBIC)

    return hairless.astype(np.uint8) 

### Example of usage (options for denoising, equalizing, resizing(224,224))
# img = preprocess("path/to/image.png", apply_eq=False, apply_denoise=False, resize=False)
