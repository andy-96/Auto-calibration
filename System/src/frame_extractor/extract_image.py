import os
import cv2
import numpy as np
from tabulate import tabulate
import sys
EXTS = {'jpg', 'jpeg', 'png'}
class VideoReader:
    def __init__(self, video_dir):
        self.__reader = cv2.VideoCapture(video_dir)

        # video name
        self.__file_name = video_dir.split('/')[-1]
        self.__video_name = self.__file_name.split('.')[0]

        # FPS
        self.__fps = self.__reader.get(cv2.CAP_PROP_FPS)
        # number of frames
        self.__frame_cnt = self.__count_frames() # int(self.__reader.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame size
        self.__frame_size = self.__reader.get(cv2.CAP_PROP_FRAME_WIDTH), self.__reader.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # summary
        print(tabulate([["name", "fps", "frame cnt", "frame size"], [self.__file_name, self.__fps, self.__frame_cnt, self.__frame_size]], headers="firstrow", tablefmt="psql"))

    def __count_frames(self):
        self.__reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.__reader.read()
        cnt = 0
        while ret:
            cnt += 1
            ret, frame = self.__reader.read()
        self.__reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cnt

    def gen_frames_by_num(self, num_frame, save_dir, img_ext='png'):
        if not isinstance(num_frame, int):
            print(f"[VideoReader.gen_frames_by_num] Error: invalid parameter {num_frame}, int required!")
            return
        if not img_ext.lower() in EXTS:
            print(f"[VideoReader.gen_frames_by_num] Error: invalid image extension {img_ext}!")
            return
        n = num_frame
        if num_frame > self.__frame_cnt:
            print(f"[VideoReader.gen_frames_by_num] required frame number {num_frame} exceeds maximum {int(self.__frame_cnt)}, return all")
            n = self.__frame_cnt
        if num_frame <= 0:
            print(f"[VideoReader.gen_frames_by_num] not requiring any frame, return all {int(self.__frame_cnt)}")
            n = self.__frame_cnt

        self.__reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        period = int(self.__frame_cnt // n)
        for i in range(1, self.__frame_cnt+1, 1):
            ret, frame = self.__reader.read()
            if i % period == 0:
                cv2.imwrite(f'{save_dir}/{self.__video_name}_{int(i//period)}.{img_ext.lower()}', frame)

        print(f"[{self.__file_name}] {n} frames generated in {save_dir}")

    def gen_frames_by_period(self, period, save_dir, img_ext='png'):
        if not isinstance(period, int):
            print(f"[VideoReader.gen_frames_by_period] Error: invalid parameter {period}, int required!")
            return
        if not img_ext.lower() in EXTS:
            print(f"VideoReader.gen_frames_by_period Error: invalid image extension {img_ext}!")
            return
        p = period
        if period > self.__frame_cnt:
            print(f"[VideoReader.gen_frames_by_period] required period {period} exceeds maximum number of frames{int(self.__frame_cnt)}, return the last frame")
            p = self.__frame_cnt
        if period <= 0:
            print(f"[VideoReader.gen_frames_by_period] required period {period} invalid, return all {int(self.__frame_cnt)} frames")
            p = 1

        for i in range(1, self.__frame_cnt+1, 1):
            ret, frame = self.__reader.read()
            if i % p == 0:
                cv2.imwrite(f'{save_dir}/{self.__video_name}_{int(i//p)}.{img_ext.lower()}', frame)

        print(f"[{self.__file_name}] {int(self.__frame_cnt / p)} frames generated in {save_dir}")


TAGS = {'mp4', 'avi', 'mov', 'mpeg', 'flv', 'wmv'}
def load_video_names(video_root):
    video_dict = dict()
    if os.path.isdir(video_root):
        video_names = sorted(os.listdir(video_root))
        if '.DS_Store' in video_names:
            video_names.remove('.DS_Store')
        cnt = 0
        for video_name in video_names:
            video_path = video_root + '/' + video_name
            if os.path.isfile(video_path):
                sep = video_name.find('.', -5)
                vtag = video_name[sep+1:]
                if vtag.lower() in TAGS:
                    video_dict[video_name[:sep]] = vtag
                    cnt += 1
        print(f"[load_video_names] {cnt} videos are loaded.")
    else:
        print(f"[load_video_names] {video_root} is not a folder!")
    return video_dict

def generate_video_frames(video_root, save_root, period, img_ext='png'):
    # load video names
    video_dict = load_video_names(video_root)
    # check save root
    if not os.path.isdir(save_root):
        try: os.mkdir(save_root)
        except:
            print(f"[generate_video_frames] Error: Invalid save root: {save_root}!")
            exit(0)
    # generate frames
    for video_name, tag in video_dict.items():
        video_dir = os.path.join(video_root, f'{video_name}.{tag}')
        vr = VideoReader(video_dir)
        vr.gen_frames_by_period(period, save_root, img_ext)
        print("")

if __name__ == "__main__":
    video_root = sys.argv[1]
    save_root = sys.argv[2]
    period = int(sys.argv[3])
    img_ext = sys.argv[4]
    generate_video_frames(video_root, save_root, period, img_ext)

