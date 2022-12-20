import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ffmpeg_dir", type=str, default="/usr/bin", help='path to ffmpeg.exe')
#parser.add_argument("--video", type=str, default='/mnt/sdc/TestVideos/720p_240fps_1.mov', required=True, help='path of video to be converted')
parser.add_argument("--root_folder1", type=str, default='/mnt/hdd12TB/SBS_DATA/SD/png', help='path of video to be converted')
parser.add_argument("--root_folder2", type=str, default='/mnt/hdd12TB/SBS_DATA/SD/vfi_png', help='path of video to be converted')
parser.add_argument("--root_folder3", type=str, default='/mnt/hdd12TB/SBS_DATA/SD/png', help='path of video to be converted')
parser.add_argument("--root_folder4", type=str, default='/mnt/hdd12TB/SBS_DATA/SD/png', help='path of video to be converted')
parser.add_argument("--video_folder1", type=str, default='/mnt/hdd12TB/SBS_DATA/220607_SBS_db_r1/png/1.올인/SD_AllIn_8th_1', help='path of video to be converted')
parser.add_argument("--video_folder2", type=str, default='/mnt/hdd12TB/SBS_DATA/220607_SBS_db_r1/png/test/SD_AllIn_8th_1', help='path of video to be converted')
parser.add_argument("--video_folder3", type=str, default='/mnt/hdd12TB/SBS_DATA/220607_SBS_db_r1/png/test/SD_AllIn_8th_1', help='path of video to be converted')
parser.add_argument("--video_folder4", type=str, default='/mnt/hdd12TB/SBS_DATA/220607_SBS_db_r1/png/test/SD_AllIn_8th_1', help='path of video to be converted')
# parser.add_argument("--video_name1", type=str, default='%6d.png', help='path of video to be converted')
# parser.add_argument("--video_name2", type=str, default='%6d.png', help='path of video to be converted')
# parser.add_argument("--video_name3", type=str, default='%6d.png', help='path of video to be converted')
# parser.add_argument("--video_name4", type=str, default='%6d.png', help='path of video to be converted')
parser.add_argument("--video_name1", type=str, default='*.mp4', help='path of video to be converted')
parser.add_argument("--video_name2", type=str, default='*.mp4', help='path of video to be converted')
parser.add_argument("--checkpoint", type=str, default='pretrained_ckpt/SuperSloMo.ckpt', help='path of checkpoint for pretrained model')

parser.add_argument("--fps", type=float, default=59.94, help='specify fps of output video. Default: 60.')
parser.add_argument("--out_folder", type=str, default="/mnt/hdd12TB/SBS_DATA/SD/merge", help='Specify output file name. Default: output.mp4')
args = parser.parse_args()




# def create_video_side_by_side(dir_name1, dir_name2, dir_name3, dir_name4, file_name1, file_name2, file_name3, file_name4, dir_out, out_name):
#     error = ""
#     #print('{} -r {} -i {}/%d.jpg -qscale:v 2 {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, args.output))
#     retn = os.system('{} -r {} -i {}/{} -r {} -i {}/{} -r {} -i {}/{} -r {} -i {}/{} '
#                      '-filter_complex "[0:v:0]pad=iw+10:ih:color=white[l]; [l][1:v:0]hstack[v]" -map "[v]" '
#                      '-b 50000k -vcodec libx264 {}/{}'.
#                      format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir_name1, file_name1,
#                             args.fps, dir_name2, file_name2, args.fps, dir_name3, file_name3, args.fps, dir_name4, file_name4, dir_out, out_name))
#     if retn:
#         error = "Error creating output video. Exiting."
#     return error

def create_video_side_by_side(dir_name1, dir_name2, dir_name3, dir_name4, dir_out, out_name):
    error = ""
    #print('{} -r {} -i {}/%d.jpg -qscale:v 2 {}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), args.fps, dir, args.output))
    retn = os.system('{} -i {} -i {} -i {} -i {} -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" {}/{}'.format(os.path.join(args.ffmpeg_dir, "ffmpeg"), dir_name1, dir_name2, dir_name3, dir_name4, dir_out, out_name))

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

    remove_file_name(args.root_folder2, '_pred')


    dir_list1 = []
    for (path, last_dir, files) in os.walk(args.root_folder1):
        for i in range (len(files)):
            if not last_dir:
                dir_list1.append(os.path.join(path, files[i]))
    dir_list1 = sorted(dir_list1)

    dir_list2 = []
    for (path, last_dir, files) in os.walk(args.root_folder2):
        for i in range (len(files)):
            if not last_dir:
                dir_list2.append(os.path.join(path, files[i]))
    dir_list2 = sorted(dir_list2)


    dir_list3 = []
    for (path, last_dir, files) in os.walk(args.root_folder3):
        for i in range (len(files)):
            if not last_dir:
                dir_list3.append(os.path.join(path, files[i]))
    dir_list3 = sorted(dir_list3)

    dir_list4 = []
    for (path, last_dir, files) in os.walk(args.root_folder4):
        for i in range (len(files)):
            if not last_dir:
                dir_list4.append(os.path.join(path, files[i]))
    dir_list4 = sorted(dir_list4)

    

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    for dir1 in dir_list1:
        sequence_name = dir1[dir1.rfind('/') + 1:]

        dir2 = [s for s in dir_list2 if sequence_name in s]
        dir3 = [s for s in dir_list3 if sequence_name in s]
        dir4 = [s for s in dir_list4 if sequence_name in s]

        if dir2:
            dir2 = dir2[0]
            dir3 = dir3[0]
            dir4 = dir4[0]

            # file_list1 = glob.glob(dir1 + '*.png')
            # file_list2 = glob.glob(dir2 + '*.png')
            # file_list3 = glob.glob(dir3 + '*.png')
            # file_list4 = glob.glob(dir4 + '*.png')

            file_list1 = glob.glob(dir1)
            file_list2 = glob.glob(dir2)
            file_list3 = glob.glob(dir3)
            file_list4 = glob.glob(dir4)

            if len(file_list1) == len(file_list2):
                out_file_name = sequence_name

                create_video_side_by_side(dir1, dir2, dir3, dir4, args.out_folder, out_file_name)
                print("Encoding Done!! {}".format(sequence_name))
            else:
                print("The number of files in the two folders is different!! {}".format(sequence_name))
        else:
            print("Pred directory not matching!!!! {}".format(sequence_name))
