import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BatchGenerator(keras.utils.Sequence):

    def __init__(self, video_path="./data/video/20bn-jester-v1",
                 img_size=(48, 48), 
                 batch_size=8,
                 use_padding=True):
        
        # Load CSV as pd.DataFrame
        self.df_labels = pd.read_csv("./data/labels.csv", header=None, names=["jester name"])
        self.df_labels['label'] = self.df_labels.index.to_series()
        self.df_train = pd.read_csv("./data/train.csv", sep=";", header=None, names=["frame_id", "jester name"])
        self.df_train = pd.merge(self.df_train, self.df_labels, how="left", on = "jester name")

        self.df_train = self.df_train[:10000]

        self.video_path = video_path
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
        max_frame_size = 0
        
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            video=[]
            for i, img_filename in enumerate(os.listdir(os.path.join(self.video_path, str(row["frame_id"])))):
                img_path = os.path.join(self.video_path, str(row["frame_id"]), str(img_filename))
                img_pil = Image.open(img_path).resize(self.img_size)
                img_arr = np.array(img_pil)
                video.append(img_arr)
            video = np.array(video)

            # max_frame_size            
            frame_size = len(video)
            if max_frame_size < frame_size:
                max_frame_size = frame_size

            x_batch.append(video)
            y_batch.append(row["label"])


        # Reverce list 
        x_batch_r = deepcopy(x_batch)
        x_batch_r.reverse()

        # Zero padding
        if self.use_padding:
            x_batch   = self._zero_padding(x_batch, max_frame_size)
            x_batch_r = self._zero_padding(x_batch_r, max_frame_size)

        x_batch = np.asarray(x_batch)
        x_batch = x_batch.astype('float32') / 255.0
        x_batch_r = np.asarray(x_batch_r)
        x_batch_r = x_batch_r.astype('float32') / 255.0
        # y_batch = np.asarray(y_batch)

        # videos, videos
        return x_batch, x_batch_r

    def _zero_padding(self, videos, max_frame_size):
        videos_pad=[]
        for v in videos:
            if v.shape[0] < max_frame_size:
                diff = max_frame_size - v.shape[0]
                v_pad = np.pad(v, [(0,diff),(0,0),(0,0),(0,0)], 'constant')
                videos_pad.append(v_pad)
            else:
                videos_pad.append(v)
        return videos_pad

    def __len__(self):
        """
        batch length: 1epochのバッチ数
        """
        return self.batches_per_epoch

    def __getlabel__(self, idx):
        batch_from = self.batch_size * idx
        batch_to = batch_from + self.batch_size

        if batch_to > self.num:
            batch_to = self.num

        label_batch = []
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            label_batch.append(row["label"])

        return np.array(label_batch)

    def on_epoch_end(self):
        # 1epochが終わった時の処理
        pass

class BatchSeq2SeqGenerator(keras.utils.Sequence):

    def __init__(self, video_path="./data/video/20bn-jester-v1",
                 img_size=(48, 48), 
                 batch_size=8,
                 use_padding=True):
        

        # Load CSV as pd.DataFrame
        self.df_labels = pd.read_csv("./data/labels.csv", header=None, names=["jester name"])
        self.df_labels['label'] = self.df_labels.index.to_series()
        self.df_train = pd.read_csv("./data/train.csv", sep=";", header=None, names=["frame_id", "jester name"])
        self.df_train = pd.merge(self.df_train, self.df_labels, how="left", on = "jester name")

        self.video_path = video_path
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
        max_frame_size = 0
        
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            video=[]
            for i, img_filename in enumerate(os.listdir(os.path.join(self.video_path, str(row["frame_id"])))):
                img_path = os.path.join(self.video_path, str(row["frame_id"]), str(img_filename))
                img_pil = Image.open(img_path).resize(self.img_size)
                img_arr = np.array(img_pil)
                video.append(img_arr)
            video = np.array(video)

            # max_frame_size            
            frame_size = len(video)
            if max_frame_size < frame_size:
                max_frame_size = frame_size

            x_batch.append(video)
            y_batch.append(row["label"])

        # Zero padding
        if self.use_padding:
            x_batch   = self._zero_padding(x_batch, max_frame_size)

        x_batch = np.asarray(x_batch)
        x_batch = x_batch.astype('float32') / 255.0
        # y_batch = np.asarray(y_batch)

        # videos, videos
        return x_batch, x_batch

    def _zero_padding(self, videos, max_frame_size):
        videos_pad=[]
        for v in videos:
            if v.shape[0] < max_frame_size:
                diff = max_frame_size - v.shape[0]
                v_pad = np.pad(v, [(0,diff),(0,0),(0,0),(0,0)], 'constant')
                videos_pad.append(v_pad)
            else:
                videos_pad.append(v)
        return videos_pad

    def __len__(self):
        """
        batch length: 1epochのバッチ数
        """
        return self.batches_per_epoch

    def __getlabel__(self, idx):
        batch_from = self.batch_size * idx
        batch_to = batch_from + self.batch_size

        if batch_to > self.num:
            batch_to = self.num

        label_batch = []
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            label_batch.append(row["label"])

        return np.array(label_batch)

    def on_epoch_end(self):
        # 1epochが終わった時の処理
        pass

    
    
class BatchDiffGenerator(keras.utils.Sequence):

    def __init__(self, video_path="./data/video/20bn-jester-v1",
                 img_size=(48, 48), 
                 batch_size=8,
                 use_padding=True):
        
        # Load CSV as pd.DataFrame
        self.df_labels = pd.read_csv("./data/labels.csv", header=None, names=["jester name"])
        self.df_labels['label'] = self.df_labels.index.to_series()
        self.df_train = pd.read_csv("./data/train.csv", sep=";", header=None, names=["frame_id", "jester name"])
        self.df_train = pd.merge(self.df_train, self.df_labels, how="left", on = "jester name")

        self.video_path = video_path
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

        # x_batch = []
        y_batch = []
        d_batch = []
        max_frame_size = 0
        
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            video=[] # original videos
            video_diff=[] # diffs between each continuous frame

            for i, img_filename in enumerate(os.listdir(os.path.join(self.video_path, str(row["frame_id"])))):
                img_path = os.path.join(self.video_path, str(row["frame_id"]), str(img_filename))
                img_pil = Image.open(img_path).resize(self.img_size)
                img_arr = np.array(img_pil)
                if i > 0:
                    img_diff = img_arr - img_pre
                    video_diff.append(img_diff)
                img_pre = deepcopy(img_arr)
                video.append(img_arr)
            video_diff.append(img_arr)

            # max_frame_size
            frame_size = len(video)
            if max_frame_size < frame_size:
                max_frame_size = frame_size

            video = np.array(video)
            video_diff = np.array(video_diff)
            
            # x_batch.append(video)
            d_batch.append(video_diff)
            y_batch.append(row["label"])

        # Reverce list 
        d_batch_r = deepcopy(d_batch)
        d_batch_r.reverse()

        # Zero padding
        if self.use_padding:
            # x_batch = self._zero_padding(x_batch, max_frame_size=50)
            d_batch   = self._zero_padding(d_batch,   max_frame_size)
            d_batch_r = self._zero_padding(d_batch_r, max_frame_size)

        # x_batch = np.asarray(x_batch)
        # x_batch = x_batch.astype('float32') / 255.0
        d_batch   = np.asarray(d_batch)
        d_batch   = d_batch.astype('float32') / 255.0
        d_batch_r = np.asarray(d_batch_r)
        d_batch_r = d_batch_r.astype('float32') / 255.0
        y_batch   = np.asarray(y_batch)

        # videos, videos
        return d_batch, d_batch_r
    
    def _zero_padding(self, videos, max_frame_size):
        videos_pad=[]
        for v in videos:
            if v.shape[0] < max_frame_size:
                diff = max_frame_size - v.shape[0]
                v_pad = np.pad(v, [(0,diff),(0,0),(0,0),(0,0)], 'constant')
                videos_pad.append(v_pad)
            else:
                videos_pad.append(v)
        return videos_pad

    def __len__(self):
        """
        batch length: 1epochのバッチ数
        """
        return self.batches_per_epoch

    def __getlabel__(self, idx):
        batch_from = self.batch_size * idx
        batch_to = batch_from + self.batch_size

        if batch_to > self.num:
            batch_to = self.num

        label_batch = []
        for index, row in self.df_train[batch_from:batch_to].iterrows(): 
            label_batch.append(row["label"])

        return np.array(label_batch)

    def on_epoch_end(self):
        # 1epochが終わった時の処理
        pass
