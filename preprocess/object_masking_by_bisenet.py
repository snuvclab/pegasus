# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor

import argparse
import json
import os
from typing import Any, Dict, List
import numpy as np
import sys

import torch
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import natsort 

face_parsing_path = os.path.abspath("./face-parsing.PyTorch")
sys.path.append(face_parsing_path)
from model import BiSeNet

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
save_pth = osp.join(os.path.join(face_parsing_path, 'res', 'cp'), '79999_iter.pth')
net.load_state_dict(torch.load(save_pth))
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def crop_rgb_image_using_mask(rgb_image_path, mask_image_path, output_image_path, white_bg=False):
    # Load the RGB image and the mask image
    rgb_image = cv2.imread(rgb_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE) # Ensure the mask is grayscale

    # Check the mask type
    if mask_image.dtype != np.uint8:
        mask_image = mask_image.astype(np.uint8)

    # Resize the mask to match the rgb_image if they are not of the same size
    if rgb_image.shape[:2] != mask_image.shape:  # We only check for height and width, not channels
        mask_image = cv2.resize(mask_image, (rgb_image.shape[1], rgb_image.shape[0]))

    # Apply the mask to the RGB image
    masked_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask_image)

    # Find the bounding box of the region of interest
    x, y, w, h = cv2.boundingRect(mask_image)

    # Crop the image from the original RGB image (not the masked image)
    cropped_from_original = rgb_image[y:y+h, x:x+w]
    cropped_mask = mask_image[y:y+h, x:x+w]

    # Combine the cropped original image with its mask to get the cropped section with white (or black) background
    if white_bg:
        cropped_with_bg = np.ones_like(cropped_from_original) * 255
    else:
        cropped_with_bg = np.zeros_like(cropped_from_original)
    cropped_with_bg[cropped_mask > 0] = cropped_from_original[cropped_mask > 0]

    # Create a blank image with the same dimensions as the original RGB image for the result
    if white_bg:
        result_image = np.ones_like(rgb_image) * 255  # This makes the background white
    else:
        result_image = np.zeros_like(rgb_image)

    # Copy the cropped section with white (or black) background to the result_image
    result_image[y:y+h, x:x+w] = cropped_with_bg

    # Save the result image
    cv2.imwrite(output_image_path, result_image)
    

def face_parsing(label1, label2, label3, file_name):
    with torch.no_grad():
        try:
            img = Image.open(file_name)
        except:
            print(file_name)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        if label3 is not None:
            condition = (parsing == label1) | (parsing == label2) | (parsing == label3)
        elif label2 is not None:
            condition = (parsing == label1) | (parsing == label2)
        else:
            condition = (parsing == label1)
    
        locations = np.where(condition)

        mask_by_parsing = (condition).astype(np.uint8) * 255

        if locations == []:
            print('[WARN] No object detected...')
            return []
        else:
            return mask_by_parsing, torch.tensor(list(zip(locations[1], locations[0])))

def main(args):
    label_mapping = {
        'skin': 1,
        'eyebrows': 2,       # 2,3
        'eyes': 4,           # 4,5 
        'ears': 7,           # 7,8
        'nose': 10,
        'mouth': 11,         # 11,12,13 (lips)
        'neck': 14,
        'necklace': 15,
        'cloth': 16,
        'hair': 17,
        'hat': 18,
    }

    label1 = label_mapping.get(args.label)
    if args.label == 'eyebrows':
        label2 = 3
        label3 = None
    elif args.label == 'eyes':
        label2 = 5
        label3 = None
    elif args.label == 'ears':
        label2 = 8
        label3 = None
    elif args.label == 'mouth':
        label2 = 12
        label3 = 13
    else:
        label2 = None
        label3 = None   

    input_path = os.path.join(args.base_path, args.input_dir)
    output_path = os.path.join(args.base_path, args.output_dir)
    rgb_output_path = os.path.join(args.base_path, '{}_rgb'.format(args.output_dir))

    if os.path.exists(output_path):
        import shutil
        shutil.rmtree(output_path)
    if os.path.exists(rgb_output_path):
        import shutil
        shutil.rmtree(rgb_output_path)

    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [
            f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))
        ]
        targets = [os.path.join(input_path, f) for f in targets]

    targets = natsort.natsorted(targets,reverse=False)
    os.makedirs(output_path, exist_ok=True)

    for t in tqdm(targets):
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks, parsing_result = face_parsing(label1, label2, label3, t)
        if args.kernel_size:
            masks = cv2.dilate(masks, np.ones((args.kernel_size, args.kernel_size), np.uint8), iterations=1)

        if parsing_result.shape[0] == 0:
            print(f"Could not find '{t}' as an image, skipping...")
            continue

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]

        os.makedirs(output_path, exist_ok=True)
        mask_image_path = os.path.join(output_path, base+'.png')
        cv2.imwrite(mask_image_path, masks)
        rgb_mask_image_path = os.path.join(rgb_output_path, base+'.png')
        os.makedirs(rgb_output_path, exist_ok=True)
        crop_rgb_image_using_mask(t, mask_image_path, rgb_mask_image_path, args.white_bg)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=8,
        help="The size of the kernel to use for the morphological operations.",
    )
    parser.add_argument(
        "--white_bg",
        action="store_true",
        help="Whether to use a white background for the mask or not.",
    )
    args = parser.parse_args()
    main(args)