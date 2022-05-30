import os
import sys
import math
import argparse
sys.path.append('.')
sys.path.append('..')
import cv2
import torch
import numpy as np
from skimage.color import rgb2yuv, yuv2rgb
from benchmark.yuv_frame_io import YUV_Read
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def HD_yuv2rgb(data_path, img_save=True):

    name_list = [
        (data_path + 'HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
        (data_path + 'HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
        (data_path + 'HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
        (data_path + 'HD1080p_GT/BlueSky.yuv', 1080, 1920),
        (data_path + 'HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
        (data_path + 'HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
        (data_path + 'HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
        (data_path + 'HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
        (data_path + 'HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
        (data_path + 'HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
        (data_path + 'HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
    ]

    down_dir = '../output/'
    if not os.path.exists(down_dir):
        os.makedirs(down_dir)

    for data in name_list:
        name = data[0]  #HD720p_GT/parkrun_1280x720_50.yuv
        h = data[1]     #720
        w = data[2]     #1280

        file_name = data[0][data[0].rfind('/') + 1:-4]
        out_fold_dir = '../output/HD/' + file_name

        if not os.path.exists(out_fold_dir):
            os.makedirs(out_fold_dir)

        if 'yuv' in name:
            Reader = YUV_Read(name, h, w, toRGB=True)   ##RGB값 변환
        else:
            Reader = cv2.VideoCapture(name)

        _, lastframe = Reader.read()                ##return값 self.RFB, True

        for index in range(0, 200, 1):
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



def HD_yuv2rgb_2(data_path, img_save=True):

    name_list = [
        (data_path + 'HD720p_GT/parkrun_1280x720_50.yuv', 720, 1280),
        (data_path + 'HD720p_GT/shields_1280x720_60.yuv', 720, 1280),
        (data_path + 'HD720p_GT/stockholm_1280x720_60.yuv', 720, 1280),
        (data_path + 'HD1080p_GT/BlueSky.yuv', 1080, 1920),
        (data_path + 'HD1080p_GT/Kimono1_1920x1080_24.yuv', 1080, 1920),
        (data_path + 'HD1080p_GT/ParkScene_1920x1080_24.yuv', 1080, 1920),
        (data_path + 'HD1080p_GT/sunflower_1080p25.yuv', 1080, 1920),
        (data_path + 'HD544p_GT/Sintel_Alley2_1280x544.yuv', 544, 1280),
        (data_path + 'HD544p_GT/Sintel_Market5_1280x544.yuv', 544, 1280),
        (data_path + 'HD544p_GT/Sintel_Temple1_1280x544.yuv', 544, 1280),
        (data_path + 'HD544p_GT/Sintel_Temple2_1280x544.yuv', 544, 1280),
    ]

    down_dir = '../output/'
    if not os.path.exists(down_dir):
        os.makedirs(down_dir)

    for data in name_list:
        name = data[0]  #HD720p_GT/parkrun_1280x720_50.yuv
        h = data[1]     #720
        w = data[2]     #1280

        file_name = data[0][data[0].rfind('/') + 1:-4]
        out_fold_dir = '../output/HD/sequences/' + file_name

        if not os.path.exists(out_fold_dir):
            os.makedirs(out_fold_dir)

        if 'yuv' in name:
            Reader = YUV_Read(name, h, w, toRGB=True)   ##RGB값 변환
        else:
            Reader = cv2.VideoCapture(name)

        _, lastframe = Reader.read()                ##return값 self.RFB, True

        f_list = open('../output/HD/'+ file_name + '_train_list.txt', 'w')
        for index in range(2, 100, 1):
            if 'yuv' in name:
                IMAGE1, success1 = Reader.read(index)
                gt, success = Reader.read(index+1)
                IMAGE2, success2 = Reader.read(index + 2)
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
            img1 = cv2.cvtColor(IMAGE1, cv2.COLOR_RGB2BGR)
            gt_ = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(IMAGE2, cv2.COLOR_RGB2BGR)

            result_dir = out_fold_dir + '/{:04d}/'.format(index)

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            if img_save:
                cv2.imwrite(os.path.join(result_dir + '/im1.png'.format(index)), img1)
                cv2.imwrite(os.path.join(result_dir + '/im2.png'.format(index+1)), gt_)
                cv2.imwrite(os.path.join(result_dir + '/im3.png'.format(index+2)), img2)
                f_list.write('{}/{:04d}\n'.format(file_name, index))

        print('finish', out_fold_dir)
    print('finish read yuv to rgb')



#HD_dataset rgb file -> downscale 1/2, 1/4
##rgb_data_path = '../output/HD/
def HD_rgb_down(rgb_data_path, img_save=True):

    name_list = [
        (rgb_data_path + 'parkrun_1280x720_50', 720, 1280),
        (rgb_data_path + 'shields_1280x720_60', 720, 1280),
        (rgb_data_path + 'stockholm_1280x720_60', 720, 1280),
        (rgb_data_path + 'BlueSky', 1080, 1920),
        (rgb_data_path + 'Kimono1_1920x1080_24', 1080, 1920),
        (rgb_data_path + 'ParkScene_1920x1080_24', 1080, 1920),
        (rgb_data_path + 'sunflower_1080p25', 1080, 1920),
        (rgb_data_path + 'Sintel_Alley2_1280x544', 544, 1280),
        (rgb_data_path + 'Sintel_Market5_1280x544', 544, 1280),
        (rgb_data_path + 'Sintel_Temple1_1280x544', 544, 1280),
        (rgb_data_path + 'Sintel_Temple2_1280x544', 544, 1280),
    ]

    for rgb_data in name_list:

        name = rgb_data[0]  ##ex) parkrun_1280x720_50
        h = rgb_data[1]
        w = rgb_data[2]
        file_name = rgb_data[0][rgb_data[0].rfind('/') + 1:]

        down_folder_dir_2 = '../output/HD_2/sequences/' + file_name    ##../output/HD_2/parkrun_1280x720_50/
        down_folder_dir_4 = '../output/HD_4/sequences/' + file_name    ##../output/HD_4/parkrun_1280x720_50/

        if not os.path.exists(down_folder_dir_2):
            os.makedirs(down_folder_dir_2)
        if not os.path.exists(down_folder_dir_4):
            os.makedirs(down_folder_dir_4)

        f2_list = open('../output/HD_2/hd2_test_list.txt', 'a')
        f4_list = open('../output/HD_4/hd4_test_list.txt', 'a')
        if 'Temple1' in file_name:
            for index in range(0, 47, 1):
                hd_rgb_dir = '../output/HD/' + file_name + '/{:04d}.png'.format(
                    index)  ##../output/HD/parkrun_1280x720_50/

                if '.png' in hd_rgb_dir:
                    image = cv2.imread(hd_rgb_dir, cv2.IMREAD_COLOR)

                    ##HD_dataset 1/2, 1/4 size
                    HD_resize_2 = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    HD_resize_4 = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                    if img_save:
                        cv2.imwrite(os.path.join(down_folder_dir_2 + '/{:04d}.png'.format(index)), HD_resize_2)
                        f2_list.write('{}/{:04d}\n'.format(file_name, index))
                        cv2.imwrite(os.path.join(down_folder_dir_4 + '/{:04d}.png'.format(index)), HD_resize_4)
                        f4_list.write('{}/{:04d}\n'.format(file_name, index))
        elif 'Sintel' in file_name:
            for index in range(0, 71, 1):
                hd_rgb_dir = '../output/HD/' + file_name + '/{:04d}.png'.format(index)  ##../output/HD/parkrun_1280x720_50/

                if '.png' in hd_rgb_dir:
                    image = cv2.imread(hd_rgb_dir, cv2.IMREAD_COLOR)

                    ##HD_dataset 1/2, 1/4 size
                    HD_resize_2 = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    HD_resize_4 = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                    if img_save:
                        cv2.imwrite(os.path.join(down_folder_dir_2 + '/{:04d}.png'.format(index)), HD_resize_2)
                        f2_list.write('{}/{:04d}\n'.format(file_name, index))
                        cv2.imwrite(os.path.join(down_folder_dir_4 + '/{:04d}.png'.format(index)), HD_resize_4)
                        f4_list.write('{}/{:04d}\n'.format(file_name, index))
        else:
            for index in range(0, 101, 1):
                hd_rgb_dir = '../output/HD/' + file_name + '/{:04d}.png'.format(index)  ##../output/HD/parkrun_1280x720_50/

                if '.png' in hd_rgb_dir:
                    image = cv2.imread(hd_rgb_dir, cv2.IMREAD_COLOR)

                    ##HD_dataset 1/2, 1/4 size
                    HD_resize_2 = cv2.resize(image, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                    HD_resize_4 = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

                    if img_save:
                        cv2.imwrite(os.path.join(down_folder_dir_2 + '/{:04d}.png'.format(index)), HD_resize_2)
                        f2_list.write('{}/{:04d}\n'.format(file_name, index))
                        cv2.imwrite(os.path.join(down_folder_dir_4 + '/{:04d}.png'.format(index)), HD_resize_4)
                        f4_list.write('{}/{:04d}\n'.format(file_name, index))

        print('finish_resize_2', down_folder_dir_2)
        print('finish resize_4', down_folder_dir_4)
    print('finish resize HD')

def HD_crop(rgb_data_path, img_save=True):
    name_list = [
        (rgb_data_path + 'parkrun_1280x720_50', 720, 1280),
        (rgb_data_path + 'shields_1280x720_60', 720, 1280),
        (rgb_data_path + 'stockholm_1280x720_60', 720, 1280),
        (rgb_data_path + 'BlueSky', 1080, 1920),
        (rgb_data_path + 'Kimono1_1920x1080_24', 1080, 1920),
        (rgb_data_path + 'ParkScene_1920x1080_24', 1080, 1920),
        (rgb_data_path + 'sunflower_1080p25', 1080, 1920),
        (rgb_data_path + 'Sintel_Alley2_1280x544', 544, 1280),
        (rgb_data_path + 'Sintel_Market5_1280x544', 544, 1280),
        (rgb_data_path + 'Sintel_Temple2_1280x544', 544, 1280),
        (rgb_data_path + 'Sintel_Temple1_1280x544', 544, 1280)
    ]

    for rgb_data in name_list:

        name = rgb_data[0]
        h = rgb_data[1]
        w = rgb_data[2]
        file_name = rgb_data[0][rgb_data[0].rfind('/') + 1:]

        crop_folder_dir = '../output/HD_crop/sequences/' + file_name

        if not os.path.exists(crop_folder_dir):
            os.makedirs(crop_folder_dir)
        f_crop_list = open('../output/HD_crop/'+ file_name + '_train_list.txt', 'w')
        f_total_list = open('../output/HD_crop/tri_train_list.txt', 'w')
        f2_list = open('../output/HD_2/hd2_test_list.txt', 'a')

        if h==720:

            for index in range(2, 99, 1):
                img1_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index)  ##../output/HD/parkrun_1280x720_50/
                gt_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index + 1)  ##../output/HD/parkrun_1280x720_50/
                img2_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index + 2)  ##../output/HD/parkrun_1280x720_50/

                image1 = cv2.imread(img1_rgb_dir, cv2.IMREAD_COLOR)
                gt = cv2.imread(gt_rgb_dir, cv2.IMREAD_COLOR)
                image2 = cv2.imread(img2_rgb_dir, cv2.IMREAD_COLOR)
                for crop_h in range(0, 522, 58):
                    for crop_w in range(0, 936, 104):
                        crop_img1 = image1[crop_h: crop_h + 256, crop_w: crop_w + 448]
                        gt_img = gt[crop_h: crop_h + 256, crop_w: crop_w + 448]
                        crop_img2 = image2[crop_h: crop_h + 256, crop_w: crop_w + 448]

                        output_dir = crop_folder_dir + '/{:04d}/'.format(index) + '{:04d}/'.format(
                            crop_h) + '{:04d}'.format(crop_w)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        if img_save:
                            cv2.imwrite(os.path.join(output_dir + '/im1.png'), crop_img1)
                            cv2.imwrite(os.path.join(output_dir + '/im2.png'), gt_img)
                            cv2.imwrite(os.path.join(output_dir + '/im3.png'), crop_img2)

                            f_crop_list.write('{}/{:04d}/{:04d}/{:04d}\n'.format(file_name, index, crop_h, crop_w))
                print('finsh_crop_{:04d}.png'.format(index))
            print('finish crop_{}'.format(file_name))

        elif h==1080:
            for index in range(2, 99, 1):
                img1_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index)  ##../output/HD/parkrun_1280x720_50/
                gt_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index + 1)  ##../output/HD/parkrun_1280x720_50/
                img2_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index + 2)  ##../output/HD/parkrun_1280x720_50/

                image1 = cv2.imread(img1_rgb_dir, cv2.IMREAD_COLOR)
                gt = cv2.imread(gt_rgb_dir, cv2.IMREAD_COLOR)
                image2 = cv2.imread(img2_rgb_dir, cv2.IMREAD_COLOR)
                for crop_h in range(0, 927, 103):
                    for crop_w in range(0, 1656, 184):
                        crop_img1 = image1[crop_h: crop_h + 256, crop_w: crop_w + 448]
                        gt_img = gt[crop_h: crop_h + 256, crop_w: crop_w + 448]
                        crop_img2 = image2[crop_h: crop_h + 256, crop_w: crop_w + 448]

                        output_dir = crop_folder_dir + '/{:04d}/'.format(index) + '{:04d}/'.format(
                            crop_h) + '{:04d}'.format(crop_w)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        if img_save:
                            cv2.imwrite(os.path.join(output_dir + '/im1.png'), crop_img1)
                            cv2.imwrite(os.path.join(output_dir + '/im2.png'), gt_img)
                            cv2.imwrite(os.path.join(output_dir + '/im3.png'), crop_img2)

                            f_crop_list.write('{}/{:04d}/{:04d}/{:04d}\n'.format(file_name, index, crop_h, crop_w))

                print('finsh_crop_{:04d}.png'.format(index))
            print('finish crop_{}'.format(file_name))


        else:
            for index in range(2, 69, 1):
                img1_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index)  ##../output/HD/parkrun_1280x720_50/
                gt_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index + 1)  ##../output/HD/parkrun_1280x720_50/
                img2_rgb_dir = '../output/HD_rgb/sequences/' + file_name + '/{:04d}.png'.format(
                    index + 2)  ##../output/HD/parkrun_1280x720_50/

                image1 = cv2.imread(img1_rgb_dir, cv2.IMREAD_COLOR)
                gt = cv2.imread(gt_rgb_dir, cv2.IMREAD_COLOR)
                image2 = cv2.imread(img2_rgb_dir, cv2.IMREAD_COLOR)
                for crop_h in range(0, 324, 36):
                    for crop_w in range(0, 936, 104):
                        crop_img1 = image1[crop_h: crop_h + 256, crop_w: crop_w + 448]
                        gt_img = gt[crop_h: crop_h + 256, crop_w: crop_w + 448]
                        crop_img2 = image2[crop_h: crop_h + 256, crop_w: crop_w + 448]

                        output_dir = crop_folder_dir + '/{:04d}/'.format(index) + '{:04d}/'.format(
                            crop_h) + '{:04d}'.format(crop_w)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)

                        if img_save:
                            cv2.imwrite(os.path.join(output_dir + '/im1.png'), crop_img1)
                            cv2.imwrite(os.path.join(output_dir + '/im2.png'), gt_img)
                            cv2.imwrite(os.path.join(output_dir + '/im3.png'), crop_img2)

                            f_crop_list.write('{}/{:04d}/{:04d}/{:04d}\n'.format(file_name, index, crop_h, crop_w))

                print('finsh_crop_{:04d}.png'.format(index))
            print('finish crop_{}'.format(file_name))
        f_total_list.write('{}/{:04d}/{:04d}/{:04d}\n'.format(file_name, index, crop_h, crop_w))
        print('finish all')




if __name__ == "__main__":
    HD_yuv2rgb('../data/HD_dataset/', img_save=True)
    HD_rgb_down('../output/HD/', img_save=True)
    HD_crop('../output/HD/', img_save=True)
