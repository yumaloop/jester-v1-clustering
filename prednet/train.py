import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import keras
import numpy as np

from prednet import PredNet
from datagen import PredNetBatchGenerator


model_train = PredNet(T=40, L=10, img_shape=(64, 64, 3)).build_train()
model_train.compile(optimizer='sgd', loss='mean_squared_error')
model_train.summary()


train_batch_generator = PredNetBatchGenerator(video_path="../data/video/20bn-jester-v1",
                                              img_size=(64, 64), 
                                              batch_size=2,
                                              T=40,
                                              L=10,
                                              use_padding=True)

date_string = "prednet_"+datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
os.mkdir('./log/'+date_string)
print("model logdir :", "./log/"+date_string)

callbacks=[]
callbacks.append(keras.callbacks.CSVLogger(filename='./log/'+date_string+'/metrics.csv'))
callbacks.append(keras.callbacks.ModelCheckpoint(filepath='./log/'+date_string+'/bestweights.hdf5', 
                                                    monitor='loss', 
                                                    save_best_only=True))

history= model_train.fit_generator(train_batch_generator, 
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
