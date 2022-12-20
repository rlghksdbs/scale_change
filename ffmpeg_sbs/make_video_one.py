import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="/usr/bin", help='path to ffmpeg.exe')
parser.add_argument("--root_folder", type=str, default='./output/sequences', help='path of video to be converted')
parser.add_argument("--video_folder", type=str, default='./output/sequences/', help='path of video to be converted')
parser.add_argument("--video_name", type=str, default='%4d.png', help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, default='pretrained_ckpt/SuperSloMo.ckpt', help='path of checkpoint for pretrained model')

parser.add_argument("--fps", type=float, default=120, help='specify fps of output video. Default: 60.')
parser.add_argument("--out_folder", type=str, default="./output/mp4_copy/", help='Specify output file name. Default: output.mp4')
args = parser.parse_args()


def create_video(dir_name, file_name, dir_out, out_name):
    error = ""
    print('{} -r {} -i {}/{} -qscale:v 2 {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir_name, file_name, out_name))
    retn = os.system('{} -r {} -i {}/{} -crf 17 -vcodec libx265 {}/{}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir_name, file_name, dir_out, out_name))
    if retn:
        error = "Error creating output video. Exiting."
    return error


def remove_file_name(root_dir, target_string):
    for (path, dir, files) in os.walk(root_dir):
        for filename in files:
            if target_string in filename:
                os.rename(os.path.join(path, filename), os.path.join(path, filename.replace(target_string, '')))
                print(os.path.join(path, filename), os.path.join(path, filename.replace(target_string, '')))

if __name__ == '__main__':

    dir_list = []
    for (path, last_dir, files) in os.walk(args.root_folder):
        if not last_dir:
            dir_list.append(path)
    dir_list = sorted(dir_list)

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    for dir in dir_list:
        sequence_name = dir[dir.rfind('/') + 1:]

        file_list1 = glob.glob(dir + '/*.png')

        out_file_name = sequence_name + '.mp4'

        create_video(dir, args.video_name, args.out_folder, out_file_name)
        print("Encoding Done!! {}".format(sequence_name))
