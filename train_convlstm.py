import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import datetime
import keras
import numpy as np
import pandas as pd
from datagen import BatchGenerator
from convlstm_autoencoder import ConvLSTMAutoEncoder


model = ConvLSTMAutoEncoder(input_shape=(50, 48, 48, 3))
model.compile(loss='mse', optimizer=keras.optimizers.Adam())
model.summary()

train_batch_generator = BatchGenerator(video_path="./data/video/20bn-jester-v1",
                                       img_size=(48, 48), 
                                       batch_size=4)

date_string = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')
os.mkdir('./log/'+date_string)
print("model logdir :", "./log/"+date_string)

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

