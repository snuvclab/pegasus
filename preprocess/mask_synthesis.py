import cv2
import numpy as np
import os
import natsort
import argparse
from tqdm import tqdm

def crop_rgb_image_using_mask(rgb_image_path, mask_image_path, output_image_path):
    # Load the RGB image and the mask image
    rgb_image = cv2.imread(rgb_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE) # Ensure the mask is grayscale

    # Apply the mask to the RGB image
    masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_image)

    # Find the bounding box of the region of interest
    x, y, w, h = cv2.boundingRect(mask_image)

    # Crop the image using the bounding box coordinates
    cropped_image = masked_image[y:y+h, x:x+w]

    # Create a blank image with the same dimensions as the original RGB image
    result_image = np.zeros_like(rgb_image)

    # Copy the cropped image to the blank image at the same location
    result_image[y:y+h, x:x+w] = cropped_image

    # Save the result image
    cv2.imwrite(output_image_path, result_image)


def upper_mask(input_img):
    # Threshold the image to binarize
    _, binarized = cv2.threshold(input_img, 127, 255, cv2.THRESH_BINARY)

    height, width = binarized.shape

    result = np.zeros((height, width), dtype=np.uint8)

    # For each x-coordinate, find the y-coordinate of the top-most white pixel
    for x in range(width):
        white_pixels_in_column = np.where(binarized[:, x] == 255)[0]
        
        if white_pixels_in_column.size == 0: # Skip if no white pixel found in the column
            continue

        # y = white_pixels_in_column[-1]  # Get the bottom-most white pixel's y-coordinate
        y = white_pixels_in_column[0]  # Get the top-most white pixel's y-coordinate
        
        result[:y, x] = 255  # Fill from top to y-coordinate with white
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='.')
    parser.add_argument('--mask_object_dir', type=str, default='mask_object')
    parser.add_argument('--mask_bald_dir', type=str, default='mask_bald')
    parser.add_argument('--mask_save_dir', type=str, default='mask')
    args = parser.parse_args()

    image_dim = (512, 512)
    mask_object_path = os.path.join(args.base_path, args.mask_object_dir)
    mask_object_images = natsort.natsorted(os.listdir(mask_object_path))

    mask_bald_path = os.path.join(args.base_path, args.mask_bald_dir)
    mask_save_path = os.path.join(args.base_path, args.mask_save_dir)
    if not os.path.exists(mask_save_path):
        os.makedirs(mask_save_path)

    for file in tqdm(mask_object_images, desc='[INFO] generate mask...'):
        mask_object = cv2.imread(os.path.join(mask_object_path, file), cv2.IMREAD_GRAYSCALE)
        mask_object = cv2.resize(mask_object, image_dim)
        mask_object_bool = mask_object > 127.5

        mask_object_upper = upper_mask(mask_object)
        mask_object_upper_bool = mask_object_upper > 127.5
        
        mask_bald = cv2.imread(os.path.join(mask_bald_path, file), cv2.IMREAD_GRAYSCALE)
        mask_bald = cv2.resize(mask_bald, image_dim)
        mask_bald_bool = mask_bald > 127.5

        mask_sticking_out_bool = np.logical_and(mask_bald_bool, mask_object_upper_bool)

        mask_blended = np.logical_or(np.logical_and(mask_bald_bool, ~mask_sticking_out_bool), mask_object_bool)
        mask_blended = mask_blended.astype(np.uint8) * 255
        cv2.imwrite(os.path.join(mask_save_path, file), mask_blended)