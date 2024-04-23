import cv2, json
import numpy as np
from tqdm import tqdm
import os
from skimage import io
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import natsort
from tqdm import tqdm
import imageio
import skimage
from scipy.sparse import diags, csr_matrix, linalg
import shutil
import sys
sys.path.append('./DECA')
from decalib.datasets import datasets
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.rotation_converter import batch_matrix2axis
from decalib.utils.util import tensor2image
from tqdm import tqdm
import torch
face_parsing_path = os.path.abspath("./face-parsing.PyTorch")
sys.path.append(face_parsing_path)
from model import BiSeNet
from PIL import Image
import torchvision.transforms as transforms
sys.path.append(os.path.abspath('./MODNet/'))
from demo.image_matting.colab.inference import image_matting_output_array, image_matting
import warnings
warnings.filterwarnings("ignore")

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.cuda()
save_pth = os.path.join(os.path.join(face_parsing_path, 'res', 'cp'), '79999_iter.pth')
net.load_state_dict(torch.load(save_pth))
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def compute_line_equation(p1, p2):
    """
    Compute the line equation y = mx + c passing through points p1 and p2.
    Returns the slope (m) and y-intercept (c).
    """
    x1, y1 = p1
    x2, y2 = p2

    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    return m, c

def create_mask_image(dimensions, p1, p2):
    """
    Create a mask image with given dimensions, where pixels above the line connecting p1 and p2 are set to 255 and others to 0.
    """
    m, c = compute_line_equation(p1, p2)
    mask = np.zeros(dimensions, dtype=np.uint8)
    
    height, width = dimensions

    for x in range(width):
        y_line = int(m * x + c)
        for y in range(height):
            if y < y_line:
                mask[y, x] = 255

    return mask

def crop_rgb_image_using_mask(rgb_image, mask_image):
    # Load the RGB image and the mask image
    # rgb_image = cv2.imread(rgb_image_path)
    # mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE) # Ensure the mask is grayscale
    # Check the mask type
    if mask_image.dtype != np.uint8:
        mask_image = mask_image.astype(np.uint8)
    # Resize the mask to match the rgb_image if they are not of the same size
    if rgb_image.shape != mask_image.shape:
        mask_image = cv2.resize(mask_image, (rgb_image.shape[1], rgb_image.shape[0]))
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
    # cv2.imwrite(output_image_path, result_image)
    return result_image

def face_parsing(image_array, label):
    with torch.no_grad():
        img = Image.fromarray(image_array)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        # condition = (parsing == 1) | (parsing == 7) | (parsing == 8) | (parsing == 10) | (parsing == 11) | (parsing == 12) | (parsing == 13)
        condition = (parsing == label)
        locations = np.where(condition)
        mask_by_parsing = (condition).astype(np.uint8) * 255

        if locations == []:
            print('[WARN] No object detected...')
            return []
        else:
            return mask_by_parsing, torch.tensor(list(zip(locations[1], locations[0])))

def face_multiple_parsing(image_array):
    with torch.no_grad():
        img = Image.fromarray(image_array)
        image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        condition = (parsing == 1) | (parsing == 2) | (parsing == 3) | (parsing == 4) | (parsing == 5) | (parsing == 6) | (parsing == 7) | (parsing == 8) | (parsing == 9) | (parsing == 10) | (parsing == 11) | (parsing == 12) | (parsing == 13)
        locations = np.where(condition)
        mask_by_parsing = (condition).astype(np.uint8) * 255

        if locations == []:
            print('[WARN] No object detected...')
            return []
        else:
            return mask_by_parsing, torch.tensor(list(zip(locations[1], locations[0])))
        
def extract_closed_regions(image_path, dimension):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dimension)

    # Ensure the image is binary (0 or 255)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours. RETR_EXTERNAL ensures only the external contours are found.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image with the same dimensions as the original image
    output = np.zeros_like(img)

    h, w = img.shape

    for contour in contours:
        is_boundary_touching = False
        
        for point in contour:
            x, y = point[0]
            if x == 0 or x == w-1 or y == 0 or y == h-1:
                is_boundary_touching = True
                break
        
        # If the contour doesn't touch the boundary, draw it
        if not is_boundary_touching:
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output

def extract_closed_regions_array(image_array):
    # Read the image
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # img = cv2.resize(img, dimension)

    img = image_array

    # Ensure the image is binary (0 or 255)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours. RETR_EXTERNAL ensures only the external contours are found.
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image with the same dimensions as the original image
    output = np.zeros_like(img)

    h, w = img.shape

    for contour in contours:
        is_boundary_touching = False
        
        for point in contour:
            x, y = point[0]
            if x == 0 or x == w-1 or y == 0 or y == h-1:
                is_boundary_touching = True
                break
        
        # If the contour doesn't touch the boundary, draw it
        if not is_boundary_touching:
            cv2.drawContours(output, [contour], -1, 255, thickness=cv2.FILLED)

    return output

def find_mask_center(mask):
    """Find the center of the white region in the mask."""
    y_indices, x_indices = np.where(mask == 255)
    top_left_y, top_left_x = np.min(y_indices), np.min(x_indices)
    bottom_right_y, bottom_right_x = np.max(y_indices), np.max(x_indices)
    y_center = (top_left_y + bottom_right_y) // 2
    x_center = (top_left_x + bottom_right_x) // 2
    return x_center, y_center

def naive_blending(src, dst, mask):
    src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
    
    dst[mask] = src[mask]
    return dst


def draw_below_contour(image, contours):
    for contour in contours:
        # Find the minimum y-coordinate for the contour
        min_y = min(point[0][1] for point in contour)
        
        # Fill the entire width below this y-coordinate
        image[min_y:,:] = 255
        
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--base_path', type=str,  help='.')
    parser.add_argument('--image_bald_dir', type=str, help='.')
    parser.add_argument('--image_rendering_dir', type=str, help='.')
    parser.add_argument('--mask_hat_dir', type=str, help='.')
    parser.add_argument('--mask_hat_rgb_dir', type=str, help='.')
    parser.add_argument('--output_image_dir', type=str, help='.')
    args = parser.parse_args()
	
	# NOTE Image dimensions (you can customize as needed)
    image_dim = (512, 512)				# (500, 500)

    image_bald_path = os.path.join(args.base_path, args.image_bald_dir)
    image_bald_list = natsort.natsorted(os.listdir(image_bald_path))
    image_rendering_path = os.path.join(args.base_path, args.image_rendering_dir)
    mask_hat_path = os.path.join(args.base_path, args.mask_hat_dir)
    mask_hat_rgb_path = os.path.join(args.base_path, args.mask_hat_rgb_dir)
    output_image_path = os.path.join(args.base_path, args.output_image_dir)
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    debug = -1

    for i, file in tqdm(enumerate(image_bald_list), desc='[INFO] Generate mask'):
        if debug != -1 and i > debug:
            break
            
        image_bald = cv2.imread(os.path.join(image_bald_path, os.path.basename(file)))
        image_bald = cv2.resize(image_bald, image_dim)

        image_rendering = cv2.imread(os.path.join(image_rendering_path, os.path.basename(file)))
        image_rendering = cv2.resize(image_rendering, image_dim)

        mask_hat = cv2.imread(os.path.join(mask_hat_path, os.path.basename(file)), cv2.IMREAD_GRAYSCALE)
        if mask_hat is None:
            print('[WARN] No mask found... {}'.format(os.path.join(mask_hat_path, os.path.basename(file))))
            continue
        mask_hat = cv2.resize(mask_hat, image_dim)
        mask_hat_bool = mask_hat > 127.5

        mask_hat_rgb = cv2.imread(os.path.join(mask_hat_rgb_path, os.path.basename(file)))
        mask_hat_rgb = cv2.resize(mask_hat_rgb, image_dim)

        naive_blended_image = np.copy(image_bald)
        naive_blended_image[mask_hat_bool] = mask_hat_rgb[mask_hat_bool]
        cv2.imwrite(os.path.join(output_image_path, os.path.basename(file)), naive_blended_image)