#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import argparse

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='vis_results/parsing_map_on_im.jpg', save_path_color='vis_results_color/parsing_map_on_im.jpg'):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0],                 # 0
                   [255, 85, 0],            # 1
                   [255, 170, 0],        # 2
                   [255, 0, 85],        # 3
                   [255, 0, 170],     # 4
                   [0, 255, 0],     # 5
                   [85, 255, 0],    # 6
                   [170, 255, 0],   # 7
                   [0, 255, 85],    # 8
                   [0, 255, 170],   # 9
                   [0, 0, 255],     # 10
                   [85, 0, 255],    # 11
                   [170, 0, 255],   # 12
                   [0, 85, 255],    # 13
                   [0, 170, 255],   # 14
                   [255, 255, 0],   # 15
                   [255, 255, 85],  # 16
                   [255, 255, 170], # 17
                   [255, 0, 255],   # 18    
                   [255, 85, 255],  # 19
                   [255, 170, 255], # 20
                   [0, 255, 255],   # 21
                   [85, 255, 255],  # 22
                   [170, 255, 255]] # 23

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        cv2.imwrite(save_path, vis_parsing_anno)
        cv2.imwrite(save_path_color, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    # return vis_im

def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):

    if not os.path.exists(respth):
        os.makedirs(respth)
    if not os.path.exists(respth+'_color'):
        os.makedirs(respth+'_color')

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            try:
                img = Image.open(osp.join(dspth, image_path))
            except:
                print(image_path)
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # parsing을 이용하여 좌표 얻어내는 방법
            # hat: np.where(parsing == 18)
            # locations = np.where(parsing == 18)
            # locations[1][index] = x coordinate, locations[0][index] = y coordinate
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path), save_path_color=osp.join(respth+'_color', image_path))







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dspth', type=str, default='/home/hyunsoocha/GitHub/IMavatar/data/datasets/syuka_hat/syuka_hat/hat_1/image', help='Path to images')
    parser.add_argument('--respth', type=str, default='/home/hyunsoocha/GitHub/IMavatar/data/datasets/syuka_hat/syuka_hat/hat_1/semantic', help='Path to makes')
    args = parser.parse_args()
    evaluate(dspth=args.dspth, respth=args.respth, cp='79999_iter.pth')


