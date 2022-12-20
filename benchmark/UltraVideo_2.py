import os
import sys

sys.path.append('.')
sys.path.append('..')
import cv2
import math
import torch
import argparse
import numpy as np
from torch.nn import functional as F
from RIFE.model.pytorch_msssim import ssim_matlab
from RIFE.model.RIFE import Model
from skimage.color import rgb2yuv, yuv2rgb
from RIFE.benchmark.yuv_frame_io import YUV_Read, YUV_Write
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def UltraVideo_2_test(data_path, pkl_name, img_save=False):
    model = Model()
    model.load_model('./train_log', pkl_name)
    model.eval()
    model.device()

    outpath_pkl = pkl_name[:-4]

    # Ultra Video
    name_list = [
        (data_path + '00_Beauty_3840x2160_60', 2160, 3840),
        (data_path + '01_Bosphorus_3840x2160_60', 2160, 3840),
        (data_path + '02_HoneyBee_3840x2160_60', 2160, 3840),
        (data_path + '03_Jockey_3840x2160_60', 2160, 3840),
        (data_path + '04_ReadySteadyGo_3840x2160_60', 2160, 3840),
        (data_path + '05_ShakeNDry_3840x2160_60', 2160, 3840),
        (data_path + '06_YachtRide_3840x2160_60', 2160, 3840),
    ]

    tot = 0.
    psnr_dir = '../output/RIFE/' + outpath_pkl + '/UltraVideo_2/'
    if not os.path.exists(psnr_dir):
        os.makedirs(psnr_dir)
    f_psnr = open(psnr_dir + '_avg_psnr.txt', 'w')

    tot_psnr_list = []
    for data in name_list:
        psnr_list = []
        name = data[0]
        h = data[1]
        w = data[2]
        only_name = name[:-4]

        file_name = data[0][data[0].rfind('/') + 1:]
        out_folder_dir = '../output/RIFE/' + outpath_pkl + '/UltraVideo_2/' + file_name

        if not os.path.exists(out_folder_dir):
            os.makedirs(out_folder_dir)

        f = []
        # f = open("/home/ketimk/VFI_Codes/frame_psnr.txt", 'a')
        f = open('{}.txt'.format(out_folder_dir), 'w')

        #f.open(data_path + '{}_test_list.txt'.format(file_name), 'r')
        inference_time = []
        if 'Shake' in file_name:
            for index in range(0, 148, 2):

                IMAGE1 = cv2.imread('../data/UltraVideo_2/sequences/' + file_name + '/{:04d}.png'.format(index))
                gt = cv2.imread('../data/UltraVideo_2/sequences/' + file_name + '/{:04d}.png'.format(index + 1))
                IMAGE2 = cv2.imread('../data/UltraVideo_2/sequences/' + file_name + '/{:04d}.png'.format(index + 2))

                I0 = torch.from_numpy(np.transpose(IMAGE1, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)
                I1 = torch.from_numpy(np.transpose(IMAGE2, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)

                if h == 720:
                    pad = 12
                elif h == 1080:
                    pad = 18
                else:
                    pad = 4
                pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
                I0 = pader(I0)
                I1 = pader(I1)
                with torch.no_grad():
                    start_st = time.time()
                    pred = model.inference(I0, I1)
                    torch.cuda.synchronize()
                    inference_time.append(time.time() - start_st)
                    pred = pred[:, :, pad: -pad]
                out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
                out_ = out / 255.
                gt = gt / 255.
                psnr = -10 * math.log10(((gt - out_) * (gt - out_)).mean())
                gt = (np.round(gt * 255)).astype('uint8')

                # video.write(out)
                # only Y PSNR

                # out_ = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                # gt_ = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

                if img_save:
                    cv2.imwrite(os.path.join(out_folder_dir, 'img_{:04d}-{:04d}.png'.format(index, index + 2)), out)
                    cv2.imwrite(os.path.join(out_folder_dir, 'gt_{:04d}-{:04d}.png'.format(index, index + 2)), gt)

                # frame_psnr = str(index+2)+'-' + str(index+4) + ', ' + str(psnr)
                # f.write(frame_psnr + '\n')
                # f.write('{:04d}-{:04d}, {:.3f}\n'.format(index, index + 2, psnr))
                f.write('{:.3f}\n'.format(psnr))
                psnr_list.append(psnr)
            # print(psnr_list)
            seq_mean = np.mean(psnr_list)
            time_mean = np.mean(inference_time)
            print('{}: {:.3f}  {:.1f}'.format(name, seq_mean, time_mean * 1000))
            tot += seq_mean
            f.write('avg psnr, {:.3f}\n'.format(seq_mean))
            f_psnr.write('{:.3f}\n'.format(seq_mean))
            f.close()

        else:
            for index in range(0, 298, 2):

                IMAGE1 = cv2.imread('../data/UltraVideo_2/sequences/' + file_name + '/{:04d}.png'.format(index))
                gt = cv2.imread('../data/UltraVideo_2/sequences/' + file_name + '/{:04d}.png'.format(index + 1))
                IMAGE2 = cv2.imread('../data/UltraVideo_2/sequences/' + file_name + '/{:04d}.png'.format(index + 2))

                I0 = torch.from_numpy(np.transpose(IMAGE1, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)
                I1 = torch.from_numpy(np.transpose(IMAGE2, (2, 0, 1)).astype("float32") / 255.).cuda().unsqueeze(0)

                if h == 720:
                    pad = 12
                elif h == 1080:
                    pad = 18
                else:
                    pad = 4
                pader = torch.nn.ReplicationPad2d([0, 0, pad, pad])
                I0 = pader(I0)
                I1 = pader(I1)
                with torch.no_grad():
                    start_st = time.time()
                    pred = model.inference(I0, I1)
                    torch.cuda.synchronize()
                    inference_time.append(time.time() - start_st)
                    pred = pred[:, :, pad: -pad]
                out = (np.round(pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)).astype('uint8')
                out_ = out / 255.
                gt = gt / 255.
                psnr = -10 * math.log10(((gt - out_) * (gt - out_)).mean())
                gt = (np.round(gt * 255)).astype('uint8')

                # video.write(out)
                # only Y PSNR

                # out_ = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                # gt_ = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)

                if img_save:
                    cv2.imwrite(os.path.join(out_folder_dir, 'img_{:04d}-{:04d}.png'.format(index, index + 2)), out)
                    cv2.imwrite(os.path.join(out_folder_dir, 'gt_{:04d}-{:04d}.png'.format(index, index + 2)), gt)

                # frame_psnr = str(index+2)+'-' + str(index+4) + ', ' + str(psnr)
                # f.write(frame_psnr + '\n')
                # f.write('{:04d}-{:04d}, {:.3f}\n'.format(index, index + 2, psnr))
                f.write('{:.3f}\n'.format(psnr))
                psnr_list.append(psnr)
            # print(psnr_list)
            seq_mean = np.mean(psnr_list)
            time_mean = np.mean(inference_time)
            print('{}: {:.3f}  {:.1f}'.format(name, seq_mean, time_mean * 1000))
            tot += seq_mean
            f.write('avg psnr, {:.3f}\n'.format(seq_mean))
            f_psnr.write('{:.3f}\n'.format(seq_mean))
            f.close()


    print('avg psnr', tot / len(name_list))
    f_psnr.write('{:.3f}\n'.format(tot / len(name_list)))
    f_psnr.close()


if __name__ == "__main__":
    UltraVideo_2_test('../data/Ultra_2_dataset/', 'flownet.pkl', img_save=True)