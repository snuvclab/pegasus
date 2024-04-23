import cv2
import numpy as np
from tqdm import tqdm
import os
import argparse
import natsort
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings("ignore")


def find_mask_center(mask):
    """Find the center of the white region in the mask."""
    y_indices, x_indices = np.where(mask == 255)
    top_left_y, top_left_x = np.min(y_indices), np.min(x_indices)
    bottom_right_y, bottom_right_x = np.max(y_indices), np.max(x_indices)
    y_center = (top_left_y + bottom_right_y) // 2
    x_center = (top_left_x + bottom_right_x) // 2
    return x_center, y_center


def poisson_blending(src, dst, mask):
    src = cv2.resize(src, (dst.shape[1], dst.shape[0]))

    center = find_mask_center(mask)
    result = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)    # cv2.NORMAL_CLONE or cv2.MIXED_CLONE

    return result

def naive_blending(src, dst, mask):
    src = cv2.resize(src, (dst.shape[1], dst.shape[0]))

    dst[mask] = src[mask]
    return dst


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_path', type=str,  help='.', default='/media/ssd2/hyunsoocha/GitHub/IMavatar/data/datasets/facial_features/facial_features/Ariana_Grande')
    parser.add_argument('--blend_type', type=str, help='.', default='poisson') # poisson or naive
    parser.add_argument('--image_original_dir', type=str, help='.')
    parser.add_argument('--image_rendering_dir', type=str, help='.')
    parser.add_argument('--mask_object_dir', type=str, help='.')
    parser.add_argument('--mask_object_rgb_dir', type=str, help='.')
    parser.add_argument('--mask_object_rendering_dir', type=str, help='.')
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--start_frame', type=int, help='.', default=0)
    args = parser.parse_args()
	
    dimensions = (512, 512)

    mask_object_path = os.path.join(args.base_path, args.mask_object_dir)
    mask_object_list = natsort.natsorted(os.listdir(mask_object_path))
    mask_object_rendering_path = os.path.join(args.base_path, args.mask_object_rendering_dir)
    mask_object_rgb_path = os.path.join(args.base_path, args.mask_object_rgb_dir)

    save_path = os.path.join(args.base_path, args.save_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    
    blacklist_dir = os.path.join(os.path.dirname(args.base_path), 'blacklist.json')
    with open(blacklist_dir, "r") as file:
        blacklist = json.load(file)

    for i, file in tqdm(enumerate(natsort.natsorted(mask_object_list)), desc="[INFO] Poisson Blending"):
        if i < args.start_frame:
            continue
        if os.path.basename(file) in blacklist[os.path.basename(args.base_path)]:
            continue
            
        mask_object = cv2.imread(os.path.join(mask_object_path, os.path.basename(file)), cv2.IMREAD_GRAYSCALE)
        mask_object = cv2.resize(mask_object, dimensions)
        kernel = (5, 5)
        mask_object = cv2.dilate(mask_object, kernel, iterations=1)
        mask_object_array = np.array(mask_object)
        
        image_rendering =  cv2.imread(os.path.join(args.base_path, args.image_rendering_dir, os.path.basename(file)))
        image_rendering = cv2.resize(image_rendering, dimensions)
        
        image_original = cv2.imread(os.path.join(args.base_path, args.image_original_dir, os.path.basename(file)))
        image_original = cv2.resize(image_original, dimensions)

        mask_object_rendering = cv2.imread(os.path.join(mask_object_rendering_path, os.path.basename(file)), cv2.IMREAD_GRAYSCALE)
        mask_object_rendering = cv2.resize(mask_object_rendering, dimensions)

        # remove the attributes from the original face.
        image_original = cv2.inpaint(image_original, mask_object, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        if args.blend_type == 'poisson':
            result = poisson_blending(image_rendering, image_original, mask_object_rendering)

        elif args.blend_type == 'naive':
            result = naive_blending(image_rendering, image_original, mask_object_array)

        cv2.imwrite(os.path.join(save_path, os.path.basename(file)), result)