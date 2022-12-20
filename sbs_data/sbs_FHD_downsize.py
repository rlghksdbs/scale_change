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
        for index in range(0, limit, 1):
            
            img0_path = os.path.join(input_clip_path, input_frames[index])

            img0 = cv2.imread(img0_path)
            h, w, _ = img0.shape

            img0 = cv2.resize(img0, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)


            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int(index))), img0)


        print('{} : previous frames {}'.format(input_clip, limit))



if __name__ == '__main__':
    out_root_path = '/home/mshong/sbs_data/1-5-1 (1K@24p)'
    # data_path = '../data/14_sbs_sd_analog'
    data_path = '/home/mshong/sbs_data/1-2 (4K@24p)'
    sbs_downsize(out_root_path, data_path, img_save=True)
    # sbs_60_to_24_2323(out_root_path, data_path, img_save=True)