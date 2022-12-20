import argparse
import os
import sys
sys.path.append('.')
sys.path.append('..')
import cv2
import math
import numpy as np

def patch_sampling(args):
    scale = args.scale
    patch_size = args.patch_size
    overlap = args.overlap_size
    lr_patch_size = int(patch_size / scale)
    lr_overlap = int(overlap / scale)
    HR_folder = '../data/DIV2K/DIV2K_train_HR'
    LR_folder = '../data/DIV2K/DIV2K_train_LR_bicubic/X{}'.format(scale)
    crop_hr_dir = '../data/DIV2K/crop_{}/overlap{}/DIV2K_train_HR'.format(patch_size, overlap)
    crop_lr_dir = '../data/DIV2K/crop_{}/overlap{}/DIV2K_train_LR_bicubic_X{}'.format(patch_size, overlap, scale)

    if not os.path.exists(crop_hr_dir):
        os.makedirs(crop_hr_dir)
    if not os.path.exists(crop_lr_dir):
        os.makedirs(crop_lr_dir)

    #make crop list
    hr_crop_list = open('../data/DIV2K/crop_{}/overlap{}/hr_crop_img_list.txt'.format(patch_size, overlap), 'w')
    lr_crop_list = open('../data/DIV2K/crop_{}/overlap{}/lr_crop_img_list.txt'.format(patch_size, overlap), 'w')



    ## generate dataset
    start_idx = 1
    end_idx = 801
    img_postfix = '.png'

    ## if store in ram
    hr_filenames = []
    lr_filenames = []
    for i in range(start_idx, end_idx):
        idx = str(i).zfill(4)
        hr_filename = os.path.join(HR_folder, idx + img_postfix)
        lr_filename = os.path.join(LR_folder, idx + 'x{}'.format(scale) + img_postfix)
        hr_filenames.append(hr_filename)
        lr_filenames.append(lr_filename)

    print("loading images in memory...")
    for i in range(0, 800, 1):
        hr_image = cv2.imread(hr_filenames[i], cv2.IMREAD_COLOR)
        hr_h = hr_image.shape[0]
        hr_w = hr_image.shape[1]
        lr_image = cv2.imread(lr_filenames[i], cv2.IMREAD_COLOR)
        lr_h = lr_image.shape[0]
        lr_w = lr_image.shape[1]
        idx = str(i+1).zfill(4)

        #crop_hr_image & save
        for crop_h in range(0, hr_h-patch_size, patch_size-overlap):
            for crop_w in range(0, hr_w-patch_size, patch_size-overlap):
                crop_hr_image = hr_image[crop_h:crop_h+patch_size, crop_w:crop_w+patch_size]
                cv2.imwrite(os.path.join(crop_hr_dir + '/{}_{}_{}.png'.format(idx, crop_h, crop_w)), crop_hr_image)
                hr_crop_list.write('{}_{}_{}.png\n'.format(idx, crop_h, crop_w))
        print('complete crop hr image' + crop_hr_dir + '/{}.png'.format(idx))

        #crop_lr_image & save
        for crop_h in range(0, lr_h - lr_patch_size, lr_patch_size-lr_overlap):
            for crop_w in range(0, lr_w - lr_patch_size, lr_patch_size-lr_overlap):
                crop_lr_image = lr_image[crop_h:crop_h+lr_patch_size, crop_w:crop_w+lr_patch_size]
                cv2.imwrite(os.path.join(crop_lr_dir + '/{}_{}_{}.png'.format(idx, crop_h, crop_w)), crop_lr_image)
                lr_crop_list.write('{}_{}_{}.png\n'.format(idx, crop_h, crop_w))
        print('complete crop lr image' + crop_lr_dir + '/{}.png'.format(idx))
    print("crop all image!")


def PSNR(HR_image, SR_image):
    mse = np.mean((HR_image - SR_image) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

def patch_pair_psnr(args):
    scale = args.scale
    patch_size = args.patch_size
    overlap = args.overlap_size
    psnr = args.psnr

    crop_hr_dir = '../data/DIV2K/crop_{}/overlap{}/DIV2K_train_HR/'.format(patch_size, overlap)
    crop_lr_dir = '../data/DIV2K/crop_{}/overlap{}/DIV2K_train_LR_bicubic_X{}/'.format(patch_size, overlap, scale)
    hr_crop_list = open('../data/DIV2K/crop_{}/overlap{}/hr_crop_img_list.txt'.format(patch_size, overlap), 'r')
    lr_crop_list = open('../data/DIV2K/crop_{}/overlap{}/lr_crop_img_list.txt'.format(patch_size, overlap), 'r')
    crop_hr_psnr_dir = '../data/DIV2K/crop_{}/overlap{}/psnr{}/DIV2K_train_HR/'.format(patch_size, overlap, psnr)
    crop_lr_psnr_dir = '../data/DIV2K/crop_{}/overlap{}/psnr{}/DIV2K_train_LR_bicubic_X{}/'.format(patch_size, overlap, psnr, scale)
    
    if not os.path.exists(crop_hr_psnr_dir):
        os.makedirs(crop_hr_psnr_dir)
    if not os.path.exists(crop_lr_psnr_dir):
        os.makedirs(crop_lr_psnr_dir)

    hr_crop_list_final = open('../data/DIV2K/crop_{}/overlap{}/psnr{}/hr_img_psnr_list.txt'.format(patch_size, overlap, psnr), 'w')
    lr_crop_list_final = open('../data/DIV2K/crop_{}/overlap{}/psnr{}/lr_img_psnr_list.txt'.format(patch_size, overlap, psnr), 'w')
    crop_psnr_list = open('../data/DIV2K/crop_{}/overlap{}/psnr{}/psnr.txt'.format(patch_size, overlap, psnr), 'w')
    pt_list_final = open('../data/DIV2K/crop_{}/overlap{}/psnr{}/DIV2K_train.txt'.format(patch_size, overlap, psnr), 'w')


    img_names = []
    hr_filename = []
    lr_filename = []
    for i in hr_crop_list:
        img_name = i[:-1]
        img_names.append(img_name)
        hr_img = crop_hr_dir + img_name
        hr_filename.append(hr_img)

    for i in lr_crop_list:
        img_name = i[:-1]
        lr_img = crop_lr_dir + img_name
        lr_filename.append(lr_img)

    #calculate crop lr&hr pair psnr and save
    a = len(hr_filename)
    for i in range(0, a, 1):
        hr_crop_img = cv2.imread(hr_filename[i], cv2.IMREAD_COLOR)
        lr_crop_img = cv2.imread(lr_filename[i], cv2.IMREAD_COLOR)
        lr_x3_crop_img = cv2.resize(lr_crop_img, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        psnr = PSNR(hr_crop_img, lr_x3_crop_img)
        crop_psnr_list.write('{}\t{}\n'.format(img_names[i],psnr))
        print('{}:{}\n'.format(img_names[i], psnr))

        if psnr <= args.psnr:
            cv2.imwrite(os.path.join(crop_hr_psnr_dir + img_names[i]), hr_crop_img)
            print("save hr_img")
            img_x3_name = img_names[i].replace('.png', 'x3.png')
            hr_crop_list_final.write('{}\n'.format(img_x3_name))
            cv2.imwrite(os.path.join(crop_lr_psnr_dir + img_x3_name), lr_crop_img)
            pt_name = img_names[i].replace('.png', '.pt')
            pt_list_final.write('{}\n'.format(pt_name))
            print("save lr_img")
            lr_crop_list_final.write('{}\n'.format(img_names[i]))

    print('fininsh')

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', default=3, type=int, help='upscale image size')
    parser.add_argument('--patch_size', default=360, type=int, help='sampling patch size')
    parser.add_argument('--overlap_size', default=180, type=int, help='crop overlapping size')
    parser.add_argument('--psnr', default=30, type=float, help='save sampling img under psnr')
    args = parser.parse_args()

    #patch_sampling(args)
    patch_pair_psnr(args)