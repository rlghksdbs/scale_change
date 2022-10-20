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
from PIL import Image


def SNU_test(data_path, SNU_mode):
    print('================== SNU start ==================\n')
    
    mode = SNU_mode

    out_root_path = './stack_image/RIFE_VGG/SNU'
    path = data_path
    f = open(os.path.join(path, 'test-{}.txt'.format(mode)), 'r')    

    out_folder_dir = os.path.join(out_root_path, 'snu_{}'.format(mode))
    if not os.path.exists(out_folder_dir):
        os.makedirs(out_folder_dir)


    for i in f:
        #name = str(i).strip()
        #if(len(name) <= 1):
        #    continue
        #print(path + 'target/' + name + '/im1.png')

        folder_name = []
        make_folder_GO = []
        make_folder_YOU = []
        #print(i)
        if i.rfind('/GOPR'):
            folder_name = i[i.rfind('/GOPR') +1:-12]
            make_folder_GO = os.path.join(out_folder_dir, folder_name)
            if not os.path.exists(make_folder_GO):
                os.makedirs(make_folder_GO)

        if i.rfind('/YouTube_test') >= 0:
            folder_name = i[i.rfind('/YouTube_') +1:-11]
            make_folder_YOU = os.path.join(out_folder_dir, folder_name)
            if not os.path.exists(make_folder_YOU):
                os.makedirs(make_folder_YOU)

        file_path = i.split()
        file = []
        file_name = file_path[1][file_path[1].rfind('/')+1:-4]
        file_name = os.path.join('img_' + file_name + '.png')

        img_path1 = os.path.join('/media/mshong/T7/주관적비교/RIFE/best_flownet_284_35.593/snu_{}'.format(mode), folder_name, file_name)
        img_path2 = os.path.join('/media/mshong/T7/주관적비교/Only_VGG/best_flownet_295_33.725/snu_{}'.format(mode), folder_name, file_name)
        
        image0 = cv2.imread(img_path1)
        image1 = cv2.imread(img_path2)

        img = np.hstack((image0, image1))        


        if i.rfind('/GOPR'):
            cv2.imwrite(os.path.join(make_folder_GO, '{}'.format(file_name)), img)

        if i.rfind('/YouTube_test') >= 0:
            cv2.imwrite(os.path.join(make_folder_YOU, '{}'.format(file_name)), img)


    
    print('================== SNU stop ==================\n')

if __name__ == "__main__":
    SNU_test('./data/10_SNU-FILM/', 'easy')
    SNU_test('./data/10_SNU-FILM/', 'medium')
    SNU_test('./data/10_SNU-FILM/', 'hard')
    SNU_test('./data/10_SNU-FILM/', 'extreme')
