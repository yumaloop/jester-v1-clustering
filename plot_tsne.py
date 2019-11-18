# %matplotlib inline
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import datasets
from sklearn.manifold import TSNE
from datagen import BatchGenerator
from convlstm_autoencoder import ConvLSTMAutoEncoder

bg = BatchGenerator(video_path="./data/latent", img_size=(48,48))

X_latent, _ = bg.__getitem__(1)
print(X_latent.shape)

tsne = TSNE(n_components=2, random_state=0)
X_plot = tsne.fit_transform(X_latent)


