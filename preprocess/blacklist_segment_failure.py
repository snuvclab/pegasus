import numpy as np
from skimage.metrics import structural_similarity as ssim
import json
import argparse
import os
import natsort
import cv2
from tqdm import tqdm

def mse(imageA, imageB):
    # Compute the mean squared error between two images
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def compare_images(imageA, imageB):
    # Compute MSE and SSIM
    m = mse(imageA, imageB)
    s, _ = ssim(imageA, imageB, full=True)
    return m, s

def compare_images_within_mask(imageA, imageB, mask_bool):
    # Apply mask to both images
    masked_imageA = imageA * mask_bool
    masked_imageB = imageB * mask_bool
    
    # Compute MSE and SSIM for masked images
    # m = mse(masked_imageA, masked_imageB)
    s, _ = ssim(masked_imageA, masked_imageB, full=True)
    
    # return m, s
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/ssd2/hyunsoocha/GitHub/PointAvatar/data/datasets/total_composition_Guy/total_composition_Guy')
    parser.add_argument('--subject_name', type=str, default='hat_Syuka_foxhat')
    parser.add_argument('--target_name', type=str, default='Syuka')
    parser.add_argument('--image_dir', type=str, default='image')
    parser.add_argument('--image_rendering_dir', type=str, default='image_rendering')
    parser.add_argument('--image_original_dir', type=str, default='image_original')
    parser.add_argument('--mask_dir', type=str, default='mask_original')  
    args = parser.parse_args()

    blacklist_file = os.path.join(args.base_path, "blacklist.json")
    if os.path.exists(blacklist_file):
        with open(blacklist_file, 'r') as f:
            blacklist = json.load(f)
    else:
        blacklist = {}

    image_dim = (512, 512)

    image_path = os.path.join(args.base_path, args.subject_name, args.image_dir)
    image_rendering_path = os.path.join(args.base_path, args.subject_name, args.image_rendering_dir)
    image_original_path = os.path.join(args.base_path, args.subject_name, args.image_original_dir)
    mask_path = os.path.join(args.base_path, args.subject_name, args.mask_dir)

    images = natsort.natsorted(os.listdir(image_path))

    outliers = []
    for i, file_name in tqdm(enumerate(images), desc='[INFO] SSIM comparison...'):
        mask = cv2.imread(os.path.join(mask_path, file_name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, image_dim)
        mask_bool = mask > 127.5
        image = cv2.imread(os.path.join(image_path, file_name), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_dim)
        image_rendering = cv2.imread(os.path.join(image_rendering_path, file_name), cv2.IMREAD_GRAYSCALE)
        image_rendering = cv2.resize(image_rendering, image_dim)
        image_original = cv2.imread(os.path.join(image_original_path, file_name), cv2.IMREAD_GRAYSCALE)
        image_original = cv2.resize(image_original, image_dim)

        ssim_rendering = compare_images_within_mask(image, image_rendering, mask_bool)
        ssim_original = compare_images_within_mask(image, image_original, mask_bool)

        if ssim_rendering > ssim_original:
            outliers.append(file_name)

    previous_blacklist = blacklist[args.subject_name]
    new_blacklist = natsort.natsorted(outliers)
    blacklist[args.subject_name] = natsort.natsorted(list(set(previous_blacklist + new_blacklist)))
    print('[INFO] added {} frames on the blacklist'.format(len(new_blacklist)))
    print('[INFO] available frames of {}: {}/{}'.format(args.subject_name, len(outliers), len(images)))

    # Save updated blacklist to file
    with open(blacklist_file, 'w') as f:
        json.dump(blacklist, f, indent=4)

    print(f"Updated blacklist saved to {blacklist_file}")