# %matplotlib inline
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from sklearn.manifold import TSNE
from datagen import BatchGenerator
from convlstm_autoencoder import ConvLSTMAutoEncoder


bg = BatchGenerator(video_path="./data/latent", img_size=(12,12))
# bg = BatchGenerator()
latent, _ = bg.__getitem__(1)
print(latent.shape)

# x = 
# tsne = TSNE(n_components=2, random_state=0)
# X_reduced = tsne.fit_transform(X)


