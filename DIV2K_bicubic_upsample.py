
import os
import sys
sys.path.append('.')
sys.path.append('..')
import cv2
import math
import numpy as np

def train_bicubic_upsampling():
    scales = 3
    LR_train_folder = './data/DIV2K/DIV2K/DIV2K_train_LR_bicubic/X3'
    HR_train_folder = './data/DIV2K/DIV2K/DIV2K_train_HR'

    lr_train_bicubic_dir = './DIV2K_X3_Bicubic/train'
    lr_val_bicubic_dir = './DIV2K_X3_Bicubic/val'

    bicubic_train_psnr_list = open('./train_psnr.txt', 'w')
    bicubic_val_psnr_list = open('./val_psnr.txt', 'w')


    if not os.path.exists(lr_train_bicubic_dir):
        os.makedirs(lr_train_bicubic_dir)


    ## generate dataset
    start_idx = 1
    end_idx = 901
    img_postfix = '.png'

    ## if store in ram
    hr_filenames = []
    lr_filenames = []

    for i in range(start_idx, end_idx):
        idx = str(i).zfill(4)
        hr_filename = os.path.join(HR_train_folder, idx + img_postfix)
        lr_filename = os.path.join(LR_train_folder, idx + 'x{}'.format(scales) + img_postfix)
        hr_filenames.append(hr_filename)
        lr_filenames.append(lr_filename)


    LEN = end_idx - start_idx
    print("loading images in memory...")
    train_total_psnr = 0.0
    for i in range(0, 800, 1):
        lr_image = cv2.imread(lr_filenames[i], cv2.IMREAD_COLOR)
        hr_image = cv2.imread(hr_filenames[i], cv2.IMREAD_COLOR)

        idx = str(i+1).zfill(4)
        lr_image_x3 = cv2.resize(lr_image, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_train_bicubic_dir + '/{}.png'.format(idx)), lr_image_x3)
        psnr = PSNR(hr_image, lr_image_x3)
        bicubic_train_psnr_list.write('{}\t {}\n'.format(idx, psnr))
        train_total_psnr += psnr
        print('{}\t {}\n'.format(idx, psnr))
        print('complete Bicubic lr image' + lr_train_bicubic_dir + '/{}.png'.format(idx))

    average_psnr = train_total_psnr / 800
    bicubic_train_psnr_list.write('train_average_psnr\t {}\n'.format(average_psnr))

    print("upsample all image!")


    val_total_psnr = 0.0
    for i in range(800, 900, 1):
        lr_image = cv2.imread(lr_filenames[i], cv2.IMREAD_COLOR)
        hr_image = cv2.imread(hr_filenames[i], cv2.IMREAD_COLOR)

        idx = str(i+1).zfill(4)
        lr_image_x3 = cv2.resize(lr_image, dsize=(0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(lr_val_bicubic_dir + '/{}.png'.format(idx)), lr_image_x3)
        val_psnr = PSNR(hr_image, lr_image_x3)
        val_toral_psnr += val_psnr
        bicubic_val_psnr_list.write('{}\t {}\n'.format(idx, val_psnr))
        print('{}\t {}\n'.format(idx, val_psnr))
        print('complete Bicubic lr image' + lr_val_bicubic_dir + '/{}.png'.format(idx))

    val_average_psnr = val_total_psnr / 100
    bicubic_val_psnr_list.write('val_average_psnr\t {}\n'.format(val_average_psnr))

    print("upsample all image!")



def PSNR(HR_image, SR_image):
    mse = np.mean((HR_image - SR_image) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr

if __name__ == "__main__":
    train_bicubic_upsampling()