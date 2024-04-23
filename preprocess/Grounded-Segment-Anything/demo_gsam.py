import os, sys
sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
import argparse
import os
import numpy as np
import torch
from PIL import Image
# Grounding DINO
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, load_image, predict
# segment anything
from segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
import natsort
from tqdm import tqdm
import argparse
face_parsing_path = os.path.abspath("../face-parsing.PyTorch")
sys.path.append(face_parsing_path)
from model import BiSeNet
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model 

device="cuda"

ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

sam_checkpoint = os.path.join('pretrain', 'sam_vit_h_4b8939.pth')
sam = build_sam(checkpoint=sam_checkpoint)
sam.to(device=device)
sam_predictor = SamPredictor(sam)

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.to(device)
ckpt_path = os.path.join(face_parsing_path, 'res/cp/79999_iter.pth')
net.load_state_dict(torch.load(ckpt_path))
net.eval()

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
)  

def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def pick_point(pi, label, parsing, W, H, w=512, h=512):
    mask_by_parsing = (parsing == pi).astype(np.uint8) * 255
    erode_kernel = np.ones((10, 10), np.uint8)
    mask_by_parsing = cv2.erode(mask_by_parsing, erode_kernel, iterations=1) / 255 * pi
    # index = np.where(parsing == pi)
    index = np.where(mask_by_parsing == pi)
    rand_idx = np.random.randint(0, len(index[0]))
    # WARNING: the order of x and y of the parsing is different from the cv2-image
    x, y = index[0][rand_idx] * W//w, index[1][rand_idx] * H//h
    point_coord = np.array([y, x]).reshape(1,2)
    point_label = np.array([label])
    return point_coord, point_label

def pick_multipoint(pi, label, parsing, W, H, w=512, h=512):
    mask_by_parsing = (parsing == pi).astype(np.uint8) * 255
    erode_kernel = np.ones((10, 10), np.uint8)
    mask_by_parsing = cv2.erode(mask_by_parsing, erode_kernel, iterations=1) / 255 * pi
    # index = np.where(parsing == pi)
    index = np.where(mask_by_parsing == pi)
    point_coord = np.zeros((10, 2))
    point_label = []
    for i in range(10):
        rand_idx = np.random.randint(0, len(index[0]))
        # WARNING: the order of x and y of the parsing is different from the cv2-image
        x, y = index[0][rand_idx] * W//w, index[1][rand_idx] * H//h
        point_coord[i] = np.array([y, x]).reshape(1, 2)
        point_label.append(label)
    point_label = np.array(point_label)
    return point_coord, point_label

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

def mask_generator(args, local_image_path):
    TEXT_PROMPT = args.text_prompt
    if TEXT_PROMPT == 'hair':
        image_parsing = cv2.imread(local_image_path)
        image = cv2.cvtColor(image_parsing, cv2.COLOR_BGR2RGB)
        image_parsing = cv2.resize(image_parsing, (512, 512)) # for saving memory if use SD-pipeline

        with torch.no_grad():
            img = Image.fromarray(image_parsing)
            H, W = img.size[0], img.size[1]
            img = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(img)
            img = torch.unsqueeze(img, 0)
            img = img.to(device)
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

        face_point, face_label = pick_point(1, 0, parsing, W, H)

        # Lists to store points and labels
        points_list = [face_point]
        labels_list = [face_label]

        if (parsing == 9).any():
            earrings_point, earrings_label = pick_point(9, 0, parsing, W, H)
            points_list.append(earrings_point)
            labels_list.append(earrings_label)

        if (parsing == 0).any():
            bg_point, bg_label = pick_point(0, 0, parsing, W, H)
            points_list.append(bg_point)
            labels_list.append(bg_label)

        if (parsing == 6).any():
            eyeglasses_point, eyeglasses_label = pick_point(6, 0, parsing, W, H)
            points_list.append(eyeglasses_point)
            labels_list.append(eyeglasses_label)

        if (parsing == 18).any():
            hat_point, hat_label = pick_point(18, 0, parsing, W, H)
            points_list.append(hat_point)
            labels_list.append(hat_label)

        if (parsing == 16).any():
            cloth_point, cloth_label = pick_point(16, 0, parsing, W, H)
            points_list.append(cloth_point)
            labels_list.append(cloth_label)

        if (parsing == 14).any():
            neck_point, neck_label = pick_point(14, 0, parsing, W, H)
            points_list.append(neck_point)
            labels_list.append(neck_label)

        # Use the lists to concatenate
        input_point = torch.tensor(np.concatenate(points_list, axis=0), device=device).unsqueeze(0)
        input_label = torch.tensor(np.concatenate(labels_list, axis=0), device=device).unsqueeze(0)
    else:
        input_point = None
        input_label = None

    
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.25

    image_source, image = load_image(local_image_path)

    boxes, logits, phrases = predict(
        model=groundingdino_model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB

    # NOTE RUN sam model
    # set image
    sam_predictor.set_image(image_source)

    # box: normalized box xywh -> unnormalized xyxy
    H, W, _ = image_source.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
                point_coords = input_point,
                point_labels = input_label,
                boxes = transformed_boxes,
                multimask_output = False,
            )
    if masks is None:
        return None

    # annotated_frame_with_mask = show_mask(masks[0][0].cpu(), annotated_frame)
    image_mask = masks[0][0].cpu().numpy()
    image_mask = image_mask.astype(np.uint8) * 255
    
    if TEXT_PROMPT == 'eyeglasses':
        # Define a kernel for the erosion operation
        # Adjust the size and shape of the kernel as per your needs
        kernel_size = 8
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Erode the mask
        eroded_mask = cv2.erode(image_mask, kernel, iterations=1)

        # Subtract the eroded mask from the original mask to get the frames
        final_mask = cv2.subtract(image_mask, eroded_mask)
    else:
        final_mask = image_mask

    return final_mask

if __name__ == "__main__":
    # If you have multiple GPUs, you can set the GPU to use here.
    # The default is to use the first GPU, which is usually GPU 0.
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/ssd1/hyunsoocha/GitHub/IMavatar/data/datasets/paste/paste/syuka_hat_1')
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--text_prompt', type=str, default="eyeglasses")
    parser.add_argument('--start_frame', type=int, help='at least 1 (png standard)', default=0)
    parser.add_argument('--white_bg', action='store_true', help='white background')
    args = parser.parse_args()

    # image_path = os.path.join(args.base_path, 'image_original')
    image_path = os.path.join(args.base_path, args.input)
    save_path = os.path.join(args.base_path, args.output)
    mask_rgb_path = os.path.join(args.base_path, 'mask_{}_rgb'.format(args.output.replace('mask_', '')))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(mask_rgb_path):
        os.makedirs(mask_rgb_path)
    for i, file in tqdm(enumerate(natsort.natsorted(os.listdir(image_path)))):
        if i < args.start_frame:
            continue
        eyeglasses_frame = mask_generator(args, os.path.join(image_path, file))
        if eyeglasses_frame is None:
            continue
        image_mask_pil = Image.fromarray(eyeglasses_frame)
        image_mask_pil.save(os.path.join(save_path, file))
        crop_rgb_image_using_mask(os.path.join(image_path, file), os.path.join(save_path, file), os.path.join(mask_rgb_path, file), args.white_bg)