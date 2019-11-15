# %matplotlib inline
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# x = 
tsne = TSNE(n_components=2, random_state=0)
X_reduced = tsne.fit_transform(X)


