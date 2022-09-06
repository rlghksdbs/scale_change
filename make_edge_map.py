import os
import sys
import cv2
import numpy as np

import torch


def make_edge_map():
    input_root = '../gtFine_trainvaltest/gtFine/train/'
    output_root = '../edge_map/'
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    folder = []
    for folder_list in os.listdir(input_root):
        folder.append(folder_list)
        output_clip_root = os.path.join(output_root, folder_list)
        if not os.path.exists(output_clip_root):
            os.makedirs(output_clip_root)

        frames = []
        for frame in sorted([file for file in os.listdir(os.path.join(input_root, folder_list)) if file.endswith("color.png")]):
            frame_root = os.path.join(input_root, folder_list, frame)
            frames.append(frame_root)
            img_path = cv2.imread(frame_root)
            dilated_frame = frame.replace('.png', '_dilated.png')
            if img_path is None:
                print("No image input")
            kernel = np.ones((4, 4), np.uint8)
            edge_map = cv2.Canny(img_path, 50, 50)
            dilated_edge_map = cv2.dilate(edge_map, kernel, iterations=1)

            edge_map_final = cv2.imwrite(os.path.join(output_clip_root, frame), edge_map)
            edge_map_dilated_final = cv2.imwrite(os.path.join(output_clip_root, dilated_frame), dilated_edge_map)
            print("img_save\t: {}".format(os.path.join(output_clip_root, frame)))
            print("dilated_img_save\t: {}".format(os.path.join(output_clip_root, dilated_frame)))


if __name__ == '__main__':
    make_edge_map()