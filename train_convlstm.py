import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import datetime
import keras
import numpy as np
import pandas as pd

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from convlstm_autoencoder import ConvLSTMAutoEncoder
from sklearn.preprocessing import LabelEncoder


class BatchGenerator(keras.utils.Sequence):

    def __init__(self, video_path="./data/video/20bn-jester-v1",
                 img_size=(48, 48), 
                 batch_size=8,
                 use_padding=True):
        
        # Load CSV as pd.DataFrame
        self.df_train = pd.read_csv("./data/train.csv", sep=";", header=None, names=["frame_id", "jester name"])
        # Label encoding
        le = LabelEncoder()
        le = le.fit(self.df_train['jester name'])
        self.df_train['label'] = le.transform(self.df_train['jester name'])

        self.num = len(self.df_train)
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_padding = use_padding
        self.batches_per_epoch = int((self.num - 1) / batch_size) + 1

    def __getitem__(self, idx):
        """
        idx: batch id
        """
        batch_from = self.batch_size * idx
        batch_to = batch_from + self.batch_size

        if batch_to > self.num:
            batch_to = self.num

        x_batch = []
        y_batch = []
        max_frame_size = 50
        
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            video=[]
            for i, img_filename in enumerate(os.listdir("./data/video/20bn-jester-v1/"+str(row["frame_id"]))):
                img_path = "./data/video/20bn-jester-v1/"+str(row["frame_id"])+"/"+str(img_filename)
                img_pil = Image.open(img_path).resize(self.img_size)
                img_arr = np.array(img_pil)
                video.append(img_arr)
            video = np.array(video)
            
            """
            if max_frame_size < video.shape[0]:
                max_frame_size = video.shape[0]
            """

            x_batch.append(video)
            y_batch.append(row["label"])
            
        # Zero padding
        if self.use_padding:
            videos_pad=[]
            for v in x_batch:
                if v.shape[0] < max_frame_size:
                    diff = max_frame_size - v.shape[0]
                    v_pad = np.pad(v, [(0,diff),(0,0),(0,0),(0,0)], 'constant')
                    videos_pad.append(v_pad)
            x_batch = videos_pad

        x_batch = np.asarray(x_batch)
        x_batch = x_batch.astype('float32') / 255.0
        y_batch = np.asarray(y_batch)

        return x_batch, x_batch

    def __len__(self):
        """
        batch length: 1epochのバッチ数
        """
        return self.batches_per_epoch

    def on_epoch_end(self):
        # 1epochが終わった時の処理
        pass

    
model = ConvLSTMAutoEncoder(input_shape=(50, 48, 48, 3))
model.compile(loss='mse', optimizer=keras.optimizers.Adam())
model.summary()

train_batch_generator = BatchGenerator(video_path="./data/video/20bn-jester-v1",
                                       img_size=(48, 48), 
                                       batch_size=8)

date_string = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
os.mkdir('./log/'+date_string)
print(date_string)

callbacks=[]
callbacks.append(keras.callbacks.CSVLogger(filename='./log/'+date_string+'/metrics.csv'))
callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./log/'+date_string+'/bestweights.hdf5', 
                                                    monitor='loss', 
                                                    save_best_only=True))

history= model.fit_generator(train_batch_generator, 
                                  steps_per_epoch=train_batch_generator.__len__(), 
                                  epochs=100, 
                                  verbose=1, 
                                  callbacks=callbacks, 
                                  validation_data=None, 
                                  validation_steps=None, 
                                  class_weight=None, 
                                  max_queue_size=1, 
                                  workers=4,
                                  use_multiprocessing=False, 
                                  shuffle=False, 
                                  initial_epoch=0)

