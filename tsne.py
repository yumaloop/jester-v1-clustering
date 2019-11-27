import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from sklearn.manifold import TSNE
from datagen import BatchGenerator

colors = list(matplotlib.colors.CSS4_COLORS.keys())[:27]
cmap = ListedColormap(colors)

# latent
bg = BatchGenerator(video_path="./data/latent1118", img_size=(48,48), batch_size=1000)
X_latent, _ = bg.__getitem__(1)
y_label = bg.__getlabel__(1)

X_latent = X_latent.reshape(int(X_latent.shape[0]), -1)

tsne = TSNE(n_components=2, random_state=0)
X_plot = tsne.fit_transform(X_latent)

plt.figure()
plt.scatter(X_plot[:, 0], X_plot[:, 1], s=10, c=y_label, cmap=cmap)
plt.savefig('latent_tsne.png') 

# original
bg = BatchGenerator(video_path="./data/video/20bn-jester-v1", img_size=(48,48), batch_size=1000)
X_ori, _ = bg.__getitem__(1)
y_label = bg.__getlabel__(1)

X_ori = X_ori.reshape(int(X_ori.shape[0]), -1)

tsne = TSNE(n_components=2, random_state=0)
X_plot = tsne.fit_transform(X_ori)

plt.figure()
plt.scatter(X_plot[:, 0], X_plot[:, 1], s=10, c=y_label, cmap=cmap)
plt.savefig('original_tsne.png') 
