import pandas as pd


df_train = pd.read_csv("./data/train.csv", sep=";", header=None, names=["frame_id", "jester name"])

# Label encoding
le = LabelEncoder()
le = le.fit(self.df_train['jester name'])
self.df_train['label'] = le.transform(self.df_train['jester name'])

self.video_path = video_path
self.num = len(self.df_train)
self.batch_size = batch_size
