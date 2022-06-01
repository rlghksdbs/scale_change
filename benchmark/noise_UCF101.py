import os
import sys
import re
import math
import random
import argparse
from collections import defaultdict

sys.path.append('.')
sys.path.append('..')
import cv2
import torch
import numpy as np
from skimage.color import rgb2yuv, yuv2rgb
from benchmark.yuv_frame_io import YUV_Read

def make_txt_list(data_root, img_save=False):

    text_file = os.path.join(data_root, 'ucfTrainTestlist/testlist01.txt')

    out_dir = '../dataset/ucf101/' + 'output/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    f_list = open(out_dir + 'noise_testlist01.txt', 'w')

    with open(text_file, "r") as f:
        lines = f.read().splitlines()
        for line in lines:
            train_list = []

            name, file_name = line.split("/")
            #file_name2, num = file_name.split(" ")
            #file_name = file_name2


            elements = re.findall("[A-Z][^A-Z]*", name)
            renamed_action = "_".join(elements)

            file_name = file_name.replace(".avi", ".jpg")
            f_list.write('{}/{}\n'.format(renamed_action, file_name))
            #f_list.close()
    print('finish textfile')


def make_noise_image(data_path, img_save=True):

    name_list = [
        (data_path + 'output/noise_trainlist01.txt'),
        (data_path + 'output/noise_trainlist02.txt'),
        (data_path + 'output/noise_trainlist03.txt'),
        (data_path + 'output/noise_testlist01.txt'),
        (data_path + 'output/noise_testlist02.txt'),
        (data_path + 'output/noise_testlist03.txt')
    ]

    out_dir = '../dataset/ucf101/' + 'output/gauss/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for text_file in name_list:
        with open(text_file, "r") as f:
            lines = f.read().splitlines()


            for image in lines:
                name, file_name = image.split("/")
                img_dir = out_dir + name
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)

                img = cv2.imread('../dataset/ucf101/UCF-101-midframes/' + image)
                '''gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                std = 15
                img_noise = make_gaussian_noise(std, gray)'''
                img_noise = noisy("gauss", img)
                #img_noise_out = cv2.cvtColor(img_noise, cv2.COLOR_GRAY2BGR)

                if img_save:
                    cv2.imwrite(os.path.join(img_dir + '/{}'.format(file_name)), img_noise)
        print('finish', text_file)
    print('finish all')



def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out

   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy