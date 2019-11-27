import os
import cv2
from tqdm import tqdm

for frame_id in tqdm(os.listdir("./video/20bn-jester-v1/")):
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter("./video_mp4/"+str(frame_id)+".mp4", fourcc, 20.0, (48, 48))

    for i, img_file in enumerate(os.listdir("./video/20bn-jester-v1/"+str(frame_id))):
        # img = cv2.imread("image{0:05d}.jpg".format(i))
        img = cv2.imread(os.path.join("./video/20bn-jester-v1/"+str(frame_id), img_file))
        img = cv2.resize(img, (48, 48))
        video.write(img)
        video.release()
