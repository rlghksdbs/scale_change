import os
import sys
import math
sys.path.append('.')
sys.path.append('..')
import cv2
import argparse
import torch
import numpy as np
from benchmark.yuv_frame_io import YUV_Read
from skimage.color import rgb2yuv, yuv2rgb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_data_path = '../data/UltraVideo_rgb/'
def Ultra_copy(rgb_data_path, img_save=True):

    name_list = [
        (rgb_data_path + '00_Beauty_3840x2160_60', 2160, 3840),
        (rgb_data_path + '01_Bosphorus_3840x2160_60', 2160, 3840),
        (rgb_data_path + '02_HoneyBee_3840x2160_60', 2160, 3840),
        (rgb_data_path + '03_Jockey_3840x2160_60', 2160, 3840),
        (rgb_data_path + '04_ReadySteadyGo_3840x2160_60', 2160, 3840),
        (rgb_data_path + '06_YachtRide_3840x2160_60', 2160, 3840),
        (rgb_data_path + '05_ShakeNDry_3840x2160_60', 2160, 3840),
    ]

    for rgb_data in name_list:

        name = rgb_data[0]
        h = rgb_data[1]
        w = rgb_data[2]
        file_name = rgb_data[0][rgb_data[0].rfind('/') + 1:]

        copy_folder_dir = '../UltraVideo_copy/sequences/' + file_name  ##../output/UltraVideo_2/01_Bpsphorous3840x2160_60/

        if not os.path.exists(copy_folder_dir):
            os.makedirs(copy_folder_dir)

        for index in range (0, 300, 1):
            ultra_rgb_dir = '../data/UltraVideo_rgb/' + file_name + '/{:04d}.png'.format(index)

            if '.png' in ultra_rgb_dir:
                image = cv2.imread(ultra_rgb_dir, cv2.IMREAD_COLOR)

                if img_save:
                    cv2.imwrite(os.path.join(copy_folder_dir + '/{:04d}.png'.format(index*2)), image)
                    cv2.imwrite(os.path.join(copy_folder_dir + '/{:04d}.png'.format((index*2)+1)), image)


        print('finish copy', copy_folder_dir)
    print('finish copy UltraVideo')

if __name__ == "__main__":
    Ultra_copy('../data/UltraVideo_rgb/', img_save=True)