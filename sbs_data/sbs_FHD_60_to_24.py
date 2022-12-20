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


def sbs_60_to_24_2224(out_root_path, data_path, img_save=False):

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
        for index in range(0, limit, 10):
            img0_path = os.path.join(input_clip_path, input_frames[index])
            img1_path = os.path.join(input_clip_path, input_frames[index+1])
            img2_path = os.path.join(input_clip_path, input_frames[index+2])
            img3_path = os.path.join(input_clip_path, input_frames[index+3])
            img4_path = os.path.join(input_clip_path, input_frames[index+4])
            img5_path = os.path.join(input_clip_path, input_frames[index+5])
            img6_path = os.path.join(input_clip_path, input_frames[index+6])
            img7_path = os.path.join(input_clip_path, input_frames[index+7])
            img8_path = os.path.join(input_clip_path, input_frames[index+8])
            img9_path = os.path.join(input_clip_path, input_frames[index+9])


            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img3 = cv2.imread(img3_path)
            img4 = cv2.imread(img4_path)
            img5 = cv2.imread(img5_path)
            img6 = cv2.imread(img6_path)
            img7 = cv2.imread(img7_path)
            img8 = cv2.imread(img8_path)
            img9 = cv2.imread(img9_path)

            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4))), img0)
            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4 + 1))), img2)
            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4 + 2))), img4)
            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4 + 3))), img6)

        print('{} : previous frames {}'.format(input_clip, limit))


def sbs_60_to_24_2323(out_root_path, data_path, img_save=False):
    
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
        for index in range(0, limit, 10):
            img0_path = os.path.join(input_clip_path, input_frames[index])
            img1_path = os.path.join(input_clip_path, input_frames[index+1])
            img2_path = os.path.join(input_clip_path, input_frames[index+2])
            img3_path = os.path.join(input_clip_path, input_frames[index+3])
            img4_path = os.path.join(input_clip_path, input_frames[index+4])
            img5_path = os.path.join(input_clip_path, input_frames[index+5])
            img6_path = os.path.join(input_clip_path, input_frames[index+6])
            img7_path = os.path.join(input_clip_path, input_frames[index+7])
            img8_path = os.path.join(input_clip_path, input_frames[index+8])
            img9_path = os.path.join(input_clip_path, input_frames[index+9])


            img0 = cv2.imread(img0_path)
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            img3 = cv2.imread(img3_path)
            img4 = cv2.imread(img4_path)
            img5 = cv2.imread(img5_path)
            img6 = cv2.imread(img6_path)
            img7 = cv2.imread(img7_path)
            img8 = cv2.imread(img8_path)
            img9 = cv2.imread(img9_path)

            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4))), img0)
            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4 + 1))), img2)
            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4 + 2))), img5)
            cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(int((index/10)*4 + 3))), img7)

        print('{} : previous frames {}'.format(input_clip, limit))

if __name__ == '__main__':
    out_root_path = '/home/mshong/sbs_UHD_png_clip_rename_60_to_24/2_2_2_4'
    # data_path = '../data/14_sbs_sd_analog'
    data_path = '/home/mshong/sbs_UHD_png_clip_rename/2_2_2_4'
    sbs_60_to_24_2224(out_root_path, data_path, img_save=True)
    # sbs_60_to_24_2323(out_root_path, data_path, img_save=True)