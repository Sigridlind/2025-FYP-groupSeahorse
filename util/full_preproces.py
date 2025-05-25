
def preprocess(image_path, mask_path, apply_eq=False, apply_denoise=False, show=True, output_size=(224, 224)):
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        raise ValueError("Image or mask could not be loaded.")

    original = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hair_type, filtered, hair_mask, img_hairless = removeHair_auto(gray, img)
    psnr, ssim = evaluate_quality(original, img_hairless, mask)
    if psnr < 30 and ssim < 0.8:
        print(f"PSNR:{psnr:.2f} and SSIM:{ssim:.3f} — not good quality")
        return None

    if apply_denoise:
        img_hairless = cv2.bilateralFilter(img_hairless, d=9, sigmaColor=75, sigmaSpace=75)

    if apply_eq:
        gray_hairless = cv2.cvtColor(img_hairless, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray_hairless)
        img_hairless = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    masked_img = cv2.bitwise_and(img_hairless, img_hairless, mask=mask)
    
    cropped_img = cut_im_by_mask(masked_img, mask)

    # Resize to fixed size (e.g. 224x224) with high-quality interpolation
    resized_img = cv2.resize(cropped_img, output_size, interpolation=cv2.INTER_CUBIC)

    # Clip values and cast to uint8 to ensure correct type
    resized_img = np.clip(resized_img, 0, 255).astype(np.uint8)

    if show:
        plt.figure(figsize=(24, 5))
        titles = [
            "Original",
            f"{hair_type.title()}-hat Filter",
            "Hair Mask",
            "Hair-Removed Full Image",
            "Cropped + Resized Lesion"
        ]
        images = [
            cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
            filtered,
            hair_mask,
            cv2.cvtColor(img_hairless, cv2.COLOR_BGR2RGB),
            cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
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

    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    return resized_img

out=preprocess(
    image_path='/Users/joakimandersen/Desktop/Projects in DS/imgs_part_1/PAT_8_15_820.png',
    mask_path='/Users/joakimandersen/Desktop/Projects in DS/lesion_masks/PAT_8_15_820_mask.png',
    apply_eq=False,
    apply_denoise=True
)
