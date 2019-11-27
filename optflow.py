import os
import cv2
import numpy as np
from tqdm import tqdm


if not os.path.exists("./data/optflow_x"):
    os.mkdir("./data/optflow_x")
if not os.path.exists("./data/optflow_y"):
    os.mkdir("./data/optflow_y")

video_dir = "./data/video/20bn-jester-v1/"
for frame_id in tqdm(os.listdir(video_dir)):
    optflow_x = []
    optflow_y = []

    for i, img_file in enumerate(os.listdir(os.path.join(video_dir, frame_id))):
        img_path = os.path.join(video_dir, frame_id, img_file)

        img_prev_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_next_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_prev_gray = cv2.resize(img_prev_gray, (48, 48))
        img_next_gray = cv2.resize(img_next_gray, (48, 48))

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(img_prev_gray, img_next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        optflow_xi = flow[:,:,0]
        optflow_yi = flow[:,:,1]

        optflow_x.append(optflow_xi)
        optflow_y.append(optflow_yi)
        
    optflow_x = np.array(optflow_x)
    optflow_y = np.array(optflow_y)

    # save fig 
    optflow_x_path = os.path.join("./data/optflow_x", frame_id+".npy")
    optflow_y_path = os.path.join("./data/optflow_y", frame_id+".npy")
    np.save(optflow_x_path, optflow_x)
    np.save(optflow_y_path, optflow_y)
