# !pip install transformers accelerate
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import torch
import cv2
import os, sys
face_parsing_path = os.path.abspath("./face-parsing.PyTorch")
sys.path.append(face_parsing_path)
from model import BiSeNet
import torchvision.transforms as transforms
from PIL import Image
import argparse
from tqdm import tqdm
import natsort

def find_mask_center(mask):
    """Find the center of the white region in the mask."""
    y_indices, x_indices = np.where(mask == 255)
    
    # If no white region is found, create a small white region in the center of the mask
    if y_indices.size == 0 or x_indices.size == 0:
        mask_center_y, mask_center_x = mask.shape[0] // 2, mask.shape[1] // 2
        # Define the size of the white region to create
        region_size = 5  # For example, a 5x5 white region
        half_size = region_size // 2
        mask[mask_center_y-half_size:mask_center_y+half_size+1, mask_center_x-half_size:mask_center_x+half_size+1] = 255
        # Update indices after modifying the mask
        y_indices, x_indices = np.where(mask == 255)

    top_left_y, top_left_x = np.min(y_indices), np.min(x_indices)
    bottom_right_y, bottom_right_x = np.max(y_indices), np.max(x_indices)
    y_center = (top_left_y + bottom_right_y) // 2
    x_center = (top_left_x + bottom_right_x) // 2
    return x_center, y_center

def poisson_blending(src, dst, mask):
    src = cv2.resize(src, (dst.shape[1], dst.shape[0]))
    # print(mask.shape)
    center = find_mask_center(mask)
    result = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)    # cv2.NORMAL_CLONE or cv2.MIXED_CLONE
    return result

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

# def face_parsing(image_array, label):
def face_parsing(img, label):
    with torch.no_grad():
        # img = Image.fromarray(image_array)
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

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

generator = torch.Generator(device="cpu").manual_seed(1)

def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', type=str, default='/media/ssd1/hyunsoocha/GitHub/IMavatar/data/datasets/total_composition/total_composition/source_bald_Airrack')
    parser.add_argument('--input_dir', type=str, default='image')
    parser.add_argument('--output_dir', type=str, default='image_bald')
    parser.add_argument('--mask_hair_dir', type=str, default='mask_hair')
    parser.add_argument('--prompt', type=str, default="bald, clean skin, smooth bald, small head, albedo")
    parser.add_argument('--negative_prompt', type=str, default="hair, wrinkles, shadow, light reflection, tattoo, sideburns, facial hair, cartoonish, abstract interpretations, hat, head coverings")
    args = parser.parse_args()
    
    input_path = os.path.join(args.base_path, args.input_dir)
    output_path = os.path.join(args.base_path, args.output_dir)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mask_hair_path = os.path.join(args.base_path, args.mask_hair_dir)
    if not os.path.exists(mask_hair_path):
        os.makedirs(mask_hair_path)
    input_images = natsort.natsorted(os.listdir(input_path))

    for i in tqdm(input_images, desc='[INFO] Inpainting for hair removal...'):
        if os.path.exists(os.path.join(output_path, i)):
            print('[WARN] Already exist... {}'.format(os.path.join(output_path, i)))
            continue
        init_image = load_image(os.path.join(input_path, i))
        init_image = init_image.resize((512, 512))

        mask_image, _ = face_parsing(init_image, 17)
        mask_image = cv2.dilate(mask_image, np.ones((20, 20), np.uint8), iterations=1)
        mask_image_bool = mask_image > 127.5
        mask_image_array = np.array(mask_image)
        mask_image = Image.fromarray(mask_image)
        mask_image.save(os.path.join(mask_hair_path, i))

        control_image = make_inpaint_condition(init_image, mask_image)

        # generate image
        image = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=20,
            generator=generator,
            eta=1.0,
            strength=1.0,
            image=init_image,
            mask_image=mask_image,
            control_image=control_image,
        ).images[0]

        init_image = np.array(init_image).astype(np.uint8)
        image_diffusion = np.array(image).astype(np.uint8)
        image_result = poisson_blending(image_diffusion, init_image, mask_image_array)
        image_result = Image.fromarray(image_result)
        image_result.save(os.path.join(output_path, i))