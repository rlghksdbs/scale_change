import os
import sys
sys.path.append('.')
sys.path.append('..')
import cv2
import math
import torch
import numpy as np
from RIFE.model.RIFE import Model
from skimage.color import rgb2yuv
from RIFE.utils.yuv_frame_io import YUV_Read
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def HD_test(data_path):
    print('================== HD start ==================\n')

    out_root_path = './stack_image/RIFE_VGG/'

    name_list = [
        (data_path + 'HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280, 100),  # name, width, height, # of max_frame
        (data_path + 'HD720p_GT/shields_1280x720_60.yuv', 720, 1280, 100),
        (data_path + 'HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280, 100),
        (data_path + 'HD1080p_GT/BlueSky.yuv', 1080, 1920, 100),
        (data_path + 'HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920, 100),
        (data_path + 'HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920, 100),
        (data_path + 'HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920, 100),
        (data_path + 'HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280, 70),
        (data_path + 'HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280, 70),
        (data_path + 'HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280, 46),
        (data_path + 'HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280, 70),
    ]
    
    tot = 0.
    tot_input4 = 0.
    
    save_dir = os.path.join(out_root_path, 'HD')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # ft = open('hd_psnr_idx.txt', "w")
    for data in name_list:
        name = data[0]
        h = data[1]
        w = data[2]
        limit = data[3]
        
        only_name = name[:-4]

        file_name = data[0][data[0].rfind('/') + 1:-4]
        out_folder_dir = os.path.join(save_dir, file_name)

        if not os.path.exists(out_folder_dir):
            os.makedirs(out_folder_dir)

        for index in range(0, limit, 2):
            img_path1 = os.path.join('/media/mshong/T7/주관적비교/RIFE/best_flownet_284_35.593/HD/{}/img_{:04d}-{:04d}.png'.format(file_name, index, index + 2))
            img_path2 = os.path.join('/media/mshong/T7/주관적비교/Only_VGG/best_flownet_295_33.725/HD/{}/img_{:04d}-{:04d}.png'.format(file_name, index, index + 2))
        
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            img = np.hstack((img1, img2))        
            cv2.imwrite(os.path.join(out_folder_dir, 'img_{:04d}-{:04d}.png'.format(index, index + 2)), img)

    print('================== HD stop ==================\n')

if __name__ == "__main__":
    HD_test('../data/09_HD_dataset/')
