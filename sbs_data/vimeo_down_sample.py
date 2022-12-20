import os
import sys
sys.path.append('.')
sys.path.append('..')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from PIL import Image
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vimeo_down():
    print('================== Vimeo90K start ==================\n')

    in_path = '/data/VFI_Database/01_vimeo_triplet'
    out_root_path = '/data/VFI_Database/vimeo_down_4/'
    f = open(os.path.join(in_path, 'tri_trainlist.txt'), 'r')

    out_folder_dir = os.path.join(out_root_path)    


    if not os.path.exists(out_folder_dir):
        os.makedirs(out_folder_dir)


    image_size_0 = []
    image_size_1 = []
    ImageSize_count = 0
    data_cnt = 0
    for i in f:
        data_cnt += 1
        name = str(i).strip()
        if len(name) <= 1:
            continue
        
        img_path = os.path.join(in_path, 'sequences', name)
        I0 = cv2.imread(img_path + '/im1.png')
        I1 = cv2.imread(img_path + '/im2.png')                            # GT
        I2 = cv2.imread(img_path + '/im3.png')                            # t+1

        os.makedirs(os.path.join(out_folder_dir, name))

        I0_ = cv2.resize(I0, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        I1_ = cv2.resize(I0, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        I2_ = cv2.resize(I0, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(out_folder_dir, name, 'im1.png'), I0_)
        cv2.imwrite(os.path.join(out_folder_dir, name, 'im2.png'), I1_)
        cv2.imwrite(os.path.join(out_folder_dir, name, 'im3.png'), I2_)


        #frame_psnr.write('{}/{}, {:.3f}\n'.format(name[:-5], name[-4:], psnr))
        print('finish: {}'.format(name))
    print('================== Vimeo90K stop ==================\n')

if __name__ == '__main__':
    vimeo_down()
