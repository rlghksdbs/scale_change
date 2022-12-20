import os
import sys

sys.path.append('.')
sys.path.append('..')
import cv2
import math
import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sbs_downsize(out_root_path, data_path, img_save=False):

    input_path = data_path
    output_path = out_root_path
    # output_path = os.path.join(out_root_path, 'sbs_120fps_sd')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    inference_time = []

    # clip 단위 처리
    for input_clip in os.listdir(input_path):
        input_clip_path = os.path.join(input_path, input_clip)
        if not os.path.isdir(input_clip_path):
            continue

        out_clip_path = os.path.join(output_path, input_clip)
        if not os.path.exists(out_clip_path):
            os.makedirs(out_clip_path)
        input_frames = sorted([file for file in os.listdir(os.path.join(input_clip_path)) if file.endswith(".png")])
        limit = len(input_frames)

        # frame 단위 처리
        for index in range(0, limit, 2):
            
            img0_path = os.path.join(input_clip_path, input_frames[index])
            img1_path = os.path.join(input_clip_path, input_frames[index+1])

            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)


            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int(index/2))), img0)


        print('{} : previous frames {}'.format(input_clip, limit))



if __name__ == '__main__':
    out_root_path = '/home/mshong/sbs_UHD_total_png_12p/2_3_2_3'
    # data_path = '../data/14_sbs_sd_analog'
    data_path = '/home/mshong/sbs_UHD_total_png_24p/2_3_2_3'
    sbs_downsize(out_root_path, data_path, img_save=True)
    # sbs_60_to_24_2323(out_root_path, data_path, img_save=True)