import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import cv2
import keras
import numpy as np
import pandas as pd
from tqdm import tqdm
from datagen import BatchGenerator

# Load model
model = keras.models.load_model("./log/base/bestweights.hdf5")
model.summary()

# build model
layer_name = 'conv3d_1'
encoder = keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
encoder.summary()

bgen = BatchGenerator(video_path="./data/video/20bn-jester-v1", img_size=(48, 48), batch_size=1, use_padding=True)


df_train = pd.read_csv("./data/train.csv", sep=";", header=None, names=["frame_id", "jester name"])                                                                                            
num_train = len(df_train)

for id_ in tqdm(range(num_train)):
    video, _ = bgen.__getitem__(int(id_))
    
    frame_id = df_train.at[int(id_), "frame_id"]
    os.mkdir('./data/latent/'+str(frame_id))
    hidden_output = encoder.predict(video)

    for i in range(50):
        lat_img = hidden_output[0][i] * 255.
        lat_img = (lat_img - lat_img.min()) / (lat_img.max() - lat_img.min())
        cv2.imwrite('./data/latent/'+str(frame_id)+'/{:0=5}.jpg'.format(int(i)), lat_img)
