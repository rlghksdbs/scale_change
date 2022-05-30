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

def UltraVideo_yuv2rgb(data_path, img_save=True):
    name_list = [
        (data_path + '00_Beauty_3840x2160_60.yuv', 2160, 3840),
        (data_path + '01_Bosphorus_3840x2160_60.yuv', 2160, 3840),
        (data_path + '02_HoneyBee_3840x2160_60.yuv', 2160, 3840),
        (data_path + '03_Jockey_3840x2160_60.yuv', 2160, 3840),
        (data_path + '04_ReadySteadyGo_3840x2160_60.yuv', 2160, 3840),
        (data_path + '05_ShakeNDry_3840x2160_60.yuv', 2160, 3840),
        (data_path + '06_YachtRide_3840x2160_60.yuv', 2160, 3840),
    ]

    down_dir = '../output/'
    if not os.path.exists(down_dir):
        os.makedirs(down_dir)

    for data in name_list:
        name = data[0]
        h = data[1]
        w = data[2]

        file_name = data[0][data[0].rfind('/') + 1:-4]
        out_fold_dir = '../output/UltraVideo/' + file_name

        if not os.path.exists(out_fold_dir):
            os.makedirs(out_fold_dir)

        if 'yuv' in name:
            Reader = YUV_Read(name, h, w, toRGB=True)
        else:
            Reader = cv2.VideoCapture(name)

        _, lastframe = Reader.read()

        for index in range(0, 400, 1):
            if 'yuv' in name:
                #IMAGE1, success1 = Reader.read(index)
                gt, success = Reader.read(index)
                #IMAGE2, success2 = Reader.read(index + 2)
                if not success:
                    break
            else:
                success1, gt = Reader.read()
                success2, frame = Reader.read()
                IMAGE1 = lastframe
                IMAGE2 = frame
                lastframe = frame
                if not success2:
                    break
#IMAGE1, IMAGE2, gt ndarry 형태
            #gt = torch.from_numpy(np.transpose(gt, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)
            #I0 = torch.from_numpy(np.transpose(IMAGE1, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)
# IMAGE1, IMAGE2, gt tensor 형태
            gt_ = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

            if img_save:
                cv2.imwrite(os.path.join(out_fold_dir + '/{:04d}.png'.format(index)), gt_)


        print('finish', out_fold_dir)
    print('finish read yuv to rgb')


def Ultra_rgb_down(rgb_data_path, img_save=True):

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

        down_folder_dir_2 = '../output/UltraVideo_2/sequences/' + file_name  ##../output/UltraVideo_2/01_Bpsphorous3840x2160_60/
        down_folder_dir_4 = '../output/UltraVideo_4/sequences/' + file_name  ##../output/UltraVideo_4/01_Bpsphorous3840x2160_60/

        if not os.path.exists(down_folder_dir_2):
            os.makedirs(down_folder_dir_2)
        if not os.path.exists(down_folder_dir_4):
            os.makedirs(down_folder_dir_4)

        for index in range (0, 300, 1):
            ultra_rgb_dir = '../output/UltraVideo/' + file_name + '/{:04d}.png'.format(index)

            if '.png' in ultra_rgb_dir:
                image = cv2.imread(ultra_rgb_dir, cv2.IMREAD_COLOR)

                Ultra_resize_2 = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                Ultra_resize_4 = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                if img_save:
                    cv2.imwrite(os.path.join(down_folder_dir_2 + '/{:04d}.png'.format(index)), Ultra_resize_2)
                    cv2.imwrite(os.path.join(down_folder_dir_4 + '/{:04d}.png'.format(index)), Ultra_resize_4)

        print('finish_resize_2', down_folder_dir_2)
        print('finish resize_4', down_folder_dir_4)
    print('finish resize UltraVideo')






if __name__ == "__main__":
    UltraVideo_yuv2rgb('../data/4KUltraVideo/', img_save=True)
    Ultra_rgb_down('../output/UltraVideo/', img_save=True)