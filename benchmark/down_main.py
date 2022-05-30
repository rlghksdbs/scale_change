import argparse
import sys
sys.path.append('.')
sys.path.append('..')
from benchmark.HD import HD_yuv2rgb, HD_rgb_down, HD_crop, HD_yuv2rgb_2
from benchmark.UltraVideo import UltraVideo_yuv2rgb, Ultra_rgb_down

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #HD
    parser.add_argument('--hd_data_path', default='../data/HD_dataset/', help='HD path')
    parser.add_argument('--hd_rgb_data_path', default='../output/HD/', help='HD rgb image path')

    #UltraVideo
    parser.add_argument('--ultra_data_path', default='../data/4KUltraVideo/', help='Ultra path')
    parser.add_argument('--ultra_rgb_data_path', default='../output/UltraVideo/', help='ultra rgb image path')

    args = parser.parse_args()

    #HD_yuv2rgb(args.hd_data_path, img_save=True)
    HD_yuv2rgb_2(args.hd_data_path, img_save=True)

    #HD_rgb_down(args.hd_rgb_data_path, img_save=True)
    #HD_crop(args.hd_rgb_data_path, img_save=True)

    #UltraVideo_yuv2rgb(args.ultra_data_path, img_save=True)
    #Ultra_rgb_down(args.ultra_rgb_data_path, img_save=True)