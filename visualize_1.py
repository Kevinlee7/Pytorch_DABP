import torch
import train
import mnist
import mnistm
import model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import itertools
import os
from utils import visualize_input


source_test_loader = mnist.mnist_test_loader
target_test_loader = mnistm.mnistm_test_loader

def plot_embedding(X, y, d, training_mode):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    y = list(itertools.chain.from_iterable(y))
    y = np.asarray(y)

    plt.figure(figsize=(10, 10))
    for i in range(len(d)):  # X.shape[0] : 1024
        # plot colored number
        if d[i] == 0:
            colors = (0.0, 0.0, 1.0, 1.0)  # 蓝色
        else:
            colors = (1.0, 0.0, 0.0, 1.0)  # 红色
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=colors,
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    plt.title(f'{training_mode} Feature Embedding')

    # 显示图像而不是保存
    plt.show()


visualize_input()
