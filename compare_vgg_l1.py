import os
import cv2
import math
from PIL import Image
import numpy as np
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def Vimeo(out_folder_dir, gt_data_path, vgg_data_path, l1_data_path, img_save=True):
    in_path = '../data/01_vimeo_triplet/'
    gt_data_path = '../data/01_vimeo_triplet/tri_testlist.txt'
    f = open(gt_data_path, 'r')
    out_folder_dir = '../compare_vgg_l1/vimeo/'
    vgg_data_path = '../vgg_img/vimeo/'
    l1_data_path = '../l1_img/vimeo/'

    if not os.path.exists(out_folder_dir):
        os.makedirs(out_folder_dir)

    if img_save:
        img_folder_dir = out_folder_dir + 'imgs'
        if not os.path.exists(img_folder_dir):
            os.makedirs(img_folder_dir)

    for i in f:
        name = str(i).strip()
        if len(name) <= 1:
            continue

        img_path = os.path.join(in_path, 'sequences', name)
        vgg_path = os.path.join(vgg_data_path, 'sequences', name)
        l1_path = os.path.join(l1_data_path, 'sequences', name)

        gt = cv2.imread(img_path + '/im2.png')  # GT
        vgg = cv2.imread(vgg_path + '/im2_vgg.png')
        l1 = cv2.imread(l1_path + '/im2_l1.png')

        final_img = np.zeros((256, 448, 3))
        compare_img = np.zeros((256, 448, 1))
        for h in range(0, 256, 1):
            for w in range(0, 448, 1):
                gt_pixel = gt[h, w, :]
                vgg_pixel = vgg[h, w, :]
                l1_pixel = l1[h, w, :]

                diff_vgg = ((gt_pixel - vgg_pixel) * (gt_pixel - vgg_pixel))
                diff_vgg_mean = diff_vgg.mean()
                diff_l1 = (gt_pixel - l1_pixel) * (gt_pixel - l1_pixel)
                diff_l1_mean = diff_l1.mean()
                diff = diff_vgg_mean - diff_l1_mean
                if diff == 0:
                    final_img[h, w, :] = vgg_pixel
                    compare_img[h, w, :] = [128]
                elif diff < 0:
                    final_img[h, w, :] = vgg_pixel
                    compare_img[h, w, :] = [255]
                else:
                    final_img[h, w, :] = l1_pixel
                    compare_img[h, w, :] = [0]
        os.makedirs(os.path.join(img_folder_dir, 'sequences', name), exist_ok=True)
        cv2.imwrite(os.path.join(img_folder_dir, 'sequences', name, 'final.png'), final_img)
        cv2.imwrite(os.path.join(img_folder_dir, 'sequences', name, 'compare.png'), compare_img)
        print("finish img")
'''

def sbs_analog(out_folder_dir, gt_data_path, vgg_data_path, l1_data_path, img_save=True):

    print('#################################SBS start#################################')
    gt_data_path = '../data/14_sbs_sd_analog/'
    out_folder_dir = '../compare_vgg_l1/sbs_analog/'
    vgg_data_path = '../vgg_img/sbs_analog/'
    l1_data_path = '../l1_img/sbs_analog/'

    if not os.path.exists(out_folder_dir):
        os.makedirs(out_folder_dir)

    clip_dir = []
    for input_clip in os.listdir(gt_data_path):
        clip_dir.append(input_clip)

    for i in range(0, 18, 1):
        gt_clip_path = os.path.join(gt_data_path, clip_dir[i])
        vgg_clip_path = os.path.join(vgg_data_path, clip_dir[i])
        l1_clip_path = os.path.join(l1_data_path, clip_dir[i])

        gt_frames = sorted([file for file in os.listdir(os.path.join(gt_clip_path)) if file.endswith(".png")])
        vgg_frames = sorted([file for file in os.listdir(os.path.join(vgg_clip_path)) if file.endswith(".png")])
        l1_frames = sorted([file for file in os.listdir(os.path.join(l1_clip_path)) if file.endswith(".png")])

        limit = len(vgg_frames)

        for idx in range(0, limit-2, 2):
            I0_frames_path = os.path.join(gt_clip_path, gt_frames[idx + 0])
            gt_frames_path = os.path.join(gt_clip_path, gt_frames[idx + 1])
            I2_frames_path = os.path.join(gt_clip_path, gt_frames[idx + 2])

            vgg_frames_path = os.path.join(vgg_clip_path, vgg_frames[idx+1])
            l1_frames_path = os.path.join(l1_clip_path, l1_frames[idx+1])

            gt = cv2.imread(gt_frames_path)  # GT
            vgg = cv2.imread(vgg_frames_path)
            l1 = cv2.imread(l1_frames_path)

            frame_h = gt.shape[0]
            frame_w = gt.shape[1]

            final_img = np.zeros((frame_h, frame_w, 3))
            compare_img = np.zeros((frame_h, frame_w, 1))

            for h in range(0, frame_h, 1):
                for w in range(0, frame_w, 1):
                    # uint8 type
                    gt_pixel = gt[h, w, :]
                    vgg_pixel = vgg[h, w, :]
                    l1_pixel = l1[h, w, :]

                    # float64 type
                    gt_pixel_ = gt_pixel / 255.
                    vgg_pixel_ = vgg_pixel / 255.
                    l1_pixel_ = l1_pixel / 255.

                    # compare difference between L1&GT, VGG&GT
                    diff_vgg = (gt_pixel_ - vgg_pixel_) * (gt_pixel_ - vgg_pixel_)
                    diff_vgg_mean = math.sqrt(np.sum(diff_vgg))
                    diff_l1 = (gt_pixel_ - l1_pixel_) * (gt_pixel_ - l1_pixel_)
                    diff_l1_mean = math.sqrt(np.sum(diff_l1))
                    diff = diff_vgg_mean - diff_l1_mean
                    diff_ = np.round((diff)).astype('int8')

                    # save compare map
                    compare_img[h, w, :] = np.clip((diff_ + 128), 0, 255)

                    # save compare pixel image
                    if diff == 0:
                        final_img[h, w, :] = vgg_pixel
                    elif diff < 0:
                        final_img[h, w, :] = vgg_pixel
                    else:
                        final_img[h, w, :] = l1_pixel

            os.makedirs(os.path.join(out_folder_dir, clip_dir[i]), exist_ok=True)
            cv2.imwrite(os.path.join(out_folder_dir, clip_dir[i], '{:06d}_final.png'.format(idx+1)), final_img)
            cv2.imwrite(os.path.join(out_folder_dir, clip_dir[i], '{:06d}_compare.png'.format(idx+1)), compare_img)
            print('finish_{}/{:06d}_final.png'.format(clip_dir[i], idx+1))
    print('#################################sbs finish#################################')

def sbs_along_compare_psnr(out_folder_dir, gt_data_path, final_data_path, img_save=True):
    out_folder_dir = '../compare_vgg_l1/psnr/sbs_along'



def SNU_FILM(out_folder_dir, SNU_mode, gt_data_path, vgg_data_path, l1_data_path, img_save=True):

    print('#################################SNU-FILM-{} start#################################'.format(SNU_mode))
    mode = SNU_mode
    gt_data_path = '../data/10_SNU-FILM/'
    out_folder_dir = '../compare_vgg_l1/snu_{}/'.format(mode)
    vgg_data_path = '../vgg_img/snu_{}'.format(mode)
    l1_data_path = '../l1_img/snu_{}'.format(mode)

    if not os.path.exists(out_folder_dir):
        os.makedirs(out_folder_dir)

    f = open(os.path.join(gt_data_path, 'test-{}.txt'.format(mode)), 'r')

    for i in f:
        make_folder_GO = []
        make_folder_YOU = []
        if img_save and i.rfind('/GOPR'):
            folder_name = i[i.rfind('/GOPR') +1:-12]
            make_folder_GO = os.path.join(out_folder_dir, folder_name)
            if not os.path.exists(make_folder_GO):
                os.makedirs(make_folder_GO)

        if img_save and i.rfind('/YouTube_test') >= 0:
            folder_name = i[i.rfind('/YouTube_') +1:-11]
            make_folder_YOU = os.path.join(out_folder_dir, folder_name)
            if not os.path.exists(make_folder_YOU):
                os.makedirs(make_folder_YOU)

        file_path = i.split()
        gt_file = []
        for a in range(0, 3, 1):
            file_paths = os.path.join('../' + file_path[a])
            gt_file_paths = file_paths.replace('SNU-FILM', '10_SNU-FILM')
            gt_file.append(gt_file_paths)
        file_name = file_path[1][file_path[1].rfind('/') + 1:-4]
        vgg_file = os.path.join(vgg_data_path, folder_name, 'img_{}.png'.format(file_name))
        l1_file = os.path.join(l1_data_path, folder_name, 'img_{}.png'.format(file_name))

        I0 = cv2.imread(gt_file[0])
        gt = cv2.imread(gt_file[1])
        I2 = cv2.imread(gt_file[2])

        vgg = cv2.imread(vgg_file)
        l1 = cv2.imread(l1_file)

        frame_h = gt.shape[0]
        frame_w = gt.shape[1]

        final_img = np.zeros((frame_h, frame_w, 3))
        compare_img = np.zeros((frame_h, frame_w, 1))

        for h in range(0, frame_h, 1):
            for w in range(0, frame_w, 1):
                #uint8 type
                gt_pixel = gt[h, w, :]
                vgg_pixel = vgg[h, w, :]
                l1_pixel = l1[h, w, :]

                #float64 type
                gt_pixel_ = gt_pixel / 1.
                vgg_pixel_ = vgg_pixel / 1.
                l1_pixel_ = l1_pixel/ 1.

                #compare difference between L1&GT, VGG&GT
                diff_vgg = (gt_pixel_ - vgg_pixel_) * (gt_pixel_ - vgg_pixel_)
                diff_vgg_mean = math.sqrt(np.sum(diff_vgg))
                diff_l1 = (gt_pixel_ - l1_pixel_) * (gt_pixel_ - l1_pixel_)
                diff_l1_mean = math.sqrt(np.sum(diff_l1))
                diff = diff_vgg_mean - diff_l1_mean
                diff_ = np.round((diff)).astype('int8')

                #save compare map
                compare_img[h, w, :] = np.clip((diff_ + 128), 0, 255)

                #save compare pixel image
                if diff == 0:
                    final_img[h, w, :] = vgg_pixel
                elif diff < 0:
                    final_img[h, w, :] = vgg_pixel
                else:
                    final_img[h, w, :] = l1_pixel

        os.makedirs(os.path.join(out_folder_dir, folder_name), exist_ok=True)
        cv2.imwrite(os.path.join(out_folder_dir, folder_name, '{}_final.png'.format(file_name)), final_img)
        cv2.imwrite(os.path.join(out_folder_dir, folder_name, '{}_compare.png'.format(file_name)), compare_img)
        print('finish_{}/{}_final.png'.format(folder_name, file_name))
    print('#################################SNU-FILM-{} finish#################################'.format(SNU_mode))


def SNU_FILM_compare_psnr(out_folder_dir, SNU_mode, gt_data_path, final_data_path, img_save=True):

    mode = SNU_mode
    print('================== SNU-{} start ==================\n'.format(SNU_mode))
    out_folder_dir = '../compare_vgg_l1/psnr/snu-{}'.format(mode)
    gt_data_path = '../data/10_SNU-FILM/'
    final_data_path = '../compare_vgg_l1/snu_{}'.format(mode)
    if not os.path.exists(out_folder_dir):
        os.makedirs(out_folder_dir)

    f = open(os.path.join(gt_data_path, 'test-{}.txt'.format(mode)), 'r')
    psnr_list = []
    frame_psnr = open(os.path.join(out_folder_dir, 'frame_psnr_{}.txt'.format(mode)), 'w')

    for i in f:
        folder_name = []
        make_folder_GO = []
        make_folder_YOU = []
        if img_save and i.rfind('/GOPR'):
            folder_name = i[i.rfind('/GOPR') +1:-12]

        if img_save and i.rfind('/YouTube_test') >= 0:
            folder_name = i[i.rfind('/YouTube_') +1:-11]

        file_path = i.split()
        gt_file = []
        for a in range(0, 3, 1):
            file_paths = os.path.join('../' + file_path[a])
            gt_file_paths = file_paths.replace('SNU-FILM', '10_SNU-FILM')
            gt_file.append(gt_file_paths)
        file_name = file_path[1][file_path[1].rfind('/') + 1:-4]
        final_file = os.path.join(final_data_path, folder_name, '{}_final.png'.format(file_name))

        gt = cv2.imread(gt_file[1])
        gt = (torch.tensor(gt.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        gt = np.round((gt[0] * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.

        final = cv2.imread(final_file)
        final = (torch.tensor(final.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        final = np.round((final[0] * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.

        psnr = -10 * math.log10(((gt - final) * (gt - final)).mean())

        frame_psnr.write('{:.3f}\n'.format(psnr))
        psnr_list.append(psnr)
    psnr_avg = np.mean(psnr_list)
    frame_psnr.write('average: {:.3f}\n'.format(psnr_avg))
    frame_psnr.close()
    print('SND', SNU_mode, psnr_avg)

    print('================== SNU stop ==================\n')

if __name__ == '__main__':
    #Vimeo(out_folder_dir='/', gt_data_path='', vgg_data_path='' ,l1_data_path='', img_save=True)
    #SNU_FILM(out_folder_dir='', SNU_mode='easy', gt_data_path='', vgg_data_path='' ,l1_data_path='', img_save=True)
    #SNU_FILM(out_folder_dir='', SNU_mode='medium', gt_data_path='', vgg_data_path='' ,l1_data_path='', img_save=True)
    #SNU_FILM(out_folder_dir='', SNU_mode='hard', gt_data_path='', vgg_data_path='' ,l1_data_path='', img_save=True)
    #SNU_FILM(out_folder_dir='', SNU_mode='extreme', gt_data_path='', vgg_data_path='' ,l1_data_path='', img_save=True)
    #sbs_analog(out_folder_dir='', gt_data_path='', vgg_data_path='' ,l1_data_path='', img_save=True)

    ###compare mixed pixel psnr
    #SNU_FILM_compare_psnr(out_folder_dir='', SNU_mode='easy', gt_data_path='', final_data_path='', img_save=True)
    SNU_FILM_compare_psnr(out_folder_dir='', SNU_mode='medium', gt_data_path='', final_data_path='', img_save=True)
    #SNU_FILM_compare_psnr(out_folder_dir='', SNU_mode='hard', gt_data_path='', final_data_path='', img_save=True)
    #SNU_FILM_compare_psnr(out_folder_dir='', SNU_mode='extreme', gt_data_path='', final_data_path='', img_save=True)
