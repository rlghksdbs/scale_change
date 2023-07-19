import cv2
import torch
import os
import argparse
import natsort
import numpy as np
from tqdm import tqdm

def add_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_path', default='./')
    parser.add_argument('--output_path', default='./')

    args = parser.parse_args()

    return args


def copy_frame(args):

    input_path = args.input_path
    output_path = args.output_path

    os.makedirs(output_path, exist_ok=True)

    frame_list = natsort.natsorted(file for file in os.listdir(input_path) if file.endswith(".png"))

    frame_len = len(frame_list)

    for i in tqdm(range(frame_len), desc='copy'):
        img_path = os.path.join(input_path, frame_list[i])
        img1 = cv2.imread(img_path)

        cv2.imwrite(os.path.join(output_path, "{:06d}.png".format(i)), img1[:540, :960, :])
        # cv2.imwrite(os.path.join(output_path, "{:06d}.png".format(i*2 + 1)), img1)
        # cv2.imwrite(os.path.join(output_path, "{:06d}.png".format(i*2 + 2)), img1)






if __name__ == '__main__':
    args = add_argparse()

    copy_frame(args)
