import os
import sys
sys.path.append('.')
sys.path.append('..')
import cv2
import math
import torch
import numpy as np
from skimage.color import rgb2yuv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def UltraVideo_test(data_path):
    print('================== UltraVideo start ==================\n')

    out_root_path = './stack_image/RIFE_VGG'

    #Ultra Video
    name_list =[
        (data_path + '00_Beauty_3840x2160_60.yuv', 2160, 3840, 298),
        (data_path + '01_Bosphorus_3840x2160_60.yuv', 2160, 3840, 298),
        (data_path + '02_HoneyBee_3840x2160_60.yuv', 2160, 3840, 298),
        (data_path + '03_Jockey_3840x2160_60.yuv', 2160, 3840, 298),
        (data_path + '04_ReadySteadyGo_3840x2160_60.yuv', 2160, 3840, 298),
        (data_path + '05_ShakeNDry_3840x2160_60.yuv', 2160, 3840, 148),
        (data_path + '06_YachtRide_3840x2160_60.yuv', 2160, 3840, 298),
    ]

    save_dir = os.path.join(out_root_path, 'UltraVideo4K')    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for data in name_list:
        name = data[0]
        only_name = name[:-4]
        limit = data[3]

        file_name = data[0][data[0].rfind('/') + 1:-4]
        out_folder_dir = os.path.join(save_dir, file_name)

        if not os.path.exists(out_folder_dir):
            os.makedirs(out_folder_dir)
        

        for index in range(0, limit, 2):

            img_path1 = os.path.join('/media/mshong/T7/주관적비교/RIFE/best_flownet_284_35.593/UltraVideo4K/{}/img_out_{:04d}-{:04d}.png'.format(file_name, index, index + 2))
            img_path2 = os.path.join('/media/mshong/T7/주관적비교/Only_VGG/best_flownet_295_33.725/UltraVideo4K/{}/img_out_{:04d}-{:04d}.png'.format(file_name, index, index + 2))
        
            img1 = cv2.imread(img_path1)
            img2 = cv2.imread(img_path2)

            img = np.hstack((img1, img2))        

            
            cv2.imwrite(os.path.join(out_folder_dir, '{:04d}-{:04d}.png'.format(index, index + 2)), img)

            
    print('================== UltraVideo stop ==================\n')

if __name__ == "__main__":
    UltraVideo_test('../data/08_4KUltraVideo/')
