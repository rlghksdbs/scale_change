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
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sbs_120fps_fhd(out_root_path, data_path, pkl_name, SynNet, img_save=False):
    print('================== SBS analog start ==================\n')

    model = Model(SynNet, arbitrary=False)
    model.load_model(pkl_name)
    # model.load_model_except_moduleWord(pkl_path, pkl_name)
    model.eval()
    model.device()

    tot = 0.
    tot_input4 = 0.

    input_path = data_path
    output_path = os.path.join(out_root_path, pkl_name[:-4], 'sbs_60fps_fhd')
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
        for index in range(0, limit - 1, 1):
            img0_path = os.path.join(input_clip_path, input_frames[index])
            img2_path = os.path.join(input_clip_path, input_frames[index + 1])

            img0 = cv2.imread(img0_path)
            img2 = cv2.imread(img2_path)

            w, h, _ = img0.shape

            img0_torch = torch.from_numpy(np.transpose(img0, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)
            img2_torch = torch.from_numpy(np.transpose(img2, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)

            # padding
            if h == 720:
                pad = 24
            elif h == 1080:
                pad = 4
            else:
                pad = 12

            pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
            img0_torch_ = pader(img0_torch)
            img2_torch_ = pader(img2_torch)

            # inference
            with torch.no_grad():
                time_start = time.time()
                pred = model.inference(img0_torch_, img2_torch_)
                torch.cuda.synchronize()
                inference_time.append(time.time() - time_start)
                pred = pred[:, :, pad: -pad]

            img1_pred = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')


            if img_save:
                cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(2*index)), img0)
                cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format(2*index + 1)), img1_pred)
                cv2.imwrite(os.path.join(output_path, input_clip, '{:06d}.png'.format((2*index + 2))), img2)

        print('finish_{}'.format(input_clip))

    avg_inference_time = np.mean(inference_time) * 1000
    print('SBS avg time: {:.4f}[msec]'.format(avg_inference_time))
    return avg_inference_time


