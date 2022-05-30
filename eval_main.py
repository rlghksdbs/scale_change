import argparse
import sys
sys.path.append('.')
sys.path.append('..')
from RIFE.benchmark.HD import HD_test
from RIFE.benchmark.Vimeo90K import vimeo_test
from RIFE.benchmark.UCF101 import UCF101_test
from RIFE.benchmark.SNU import SNU_test
from RIFE.benchmark.UltraVideo import UltraVideo_test
from RIFE.benchmark.UltraVideo_2 import UltraVideo_2_test
from RIFE.benchmark.UltraVideo_4 import UltraVideo_4_test

from RIFE.benchmark.SJTU import SJTU_test
from RIFE.benchmark.TVD_4K import TVD_4K_test
from RIFE.benchmark.HD_2 import HD_2_test
from RIFE.benchmark.HD_4 import HD_4_test



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_name', default='flownet.pkl', help='dataset path')

    # HD argument
    parser.add_argument('--hd_data_path', default='../data/HD_dataset/', help='dataset path')
    parser.add_argument('--hd_2_data_path', default='../data/HD_2/', help= 'HD downsacale dataset path')
    parser.add_argument('--hd_4_data_path', default='../data/HD_4/', help= 'HD downsacale dataset path')


    # Vimeo argument
    parser.add_argument('--vimeo_model_path', default='./train_log', help="pretrained_model path")
    parser.add_argument('--vimeo_data_path', default='../data/vimeo_triplet/', help='dataset path')
    parser.add_argument('--vimeo_txt_name', default='tri_testlist.txt', help='text file path')
    parser.add_argument('--vimeo_save_img', default=False, help='save I0, I2 and output image (I1)')

    # UCF101 argument
    parser.add_argument('--ucf_data_path', default='../data/ucf101_interp_DB/', help='dataset path')

    # SNU argument
    parser.add_argument('--snu_data_path', default='../data/SNU-FILM/', help='dataset path')
    #parser.add_argument('--snu_mode', default='easy', help="easy, medium, hard, extreme")

    # UltraVideo argument
    parser.add_argument('--ultra_data_path', default='../data/4KUltraVideo/', help='dataset path')
    parser.add_argument('--ultra_2_data_path', default='../data/UltraVideo_2/', help='ultra downsacle dataset path')
    parser.add_argument('--ultra_4_data_path', default='../data/UltraVideo_4/', help='ultra downsacle dataset path')

    # SJTU argument
    parser.add_argument('--sjtu_data_path', default='../data/4K_SJTU_Media_Lab/', help='dataset path')

    # TVD argument
    parser.add_argument('--tvd_data_path', default='../data/TVD_4K/', help='dataset path')

    args = parser.parse_args()


    HD_2_test(args.hd_2_data_path, args.pkl_name, img_save=True)
    HD_4_test(args.hd_4_data_path, args.pkl_name, img_save=True)
    HD_test(args.hd_data_path, args.pkl_name, img_save=True)
    #vimeo_test(args.vimeo_data_path, args.vimeo_txt_name, args.vimeo_model_path, args.pkl_name, args.vimeo_save_img)
    #UCF101_test(args.ucf_data_path, args.pkl_name)
    #SNU_test(args.snu_data_path, args.pkl_name, 'easy', img_save=False)
    #SNU_test(args.snu_data_path, args.pkl_name, 'medium', img_save=False)
    #SNU_test(args.snu_data_path, args.pkl_name, 'hard', img_save=False)
    #SNU_test(args.snu_data_path, args.pkl_name, 'extreme', img_save=False)
    UltraVideo_test(args.ultra_data_path, args.pkl_name, img_save=True)
    UltraVideo_2_test(args.ultra_2_data_path, args.pkl_name, img_save=True)
    UltraVideo_4_test(args.ultra_4_data_path, args.pkl_name, img_save=True)

    #SJTU_test(args.sjtu_data_path, args.pkl_name)
    # TVD_4K_test(args.tvd_data_path, args.pkl_name)

    print("@@@ RIFE test complete @@@")

