import torch
import train
import mnist
import mnistm
import model
from utils import get_free_gpu, extract_features  # 假设你有一个提取特征的函数
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import itertools
import os

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

def visualize_input():
    source_test_loader = mnist.mnist_test_loader
    target_test_loader = mnistm.mnist_test_loader

    # 获取源测试样本
    source_label_list = []
    source_img_list = []
    for i, test_data in enumerate(source_test_loader):
        if i >= 16:  # 仅获取512个样本
            break
        img, label = test_data
        label = label.numpy()
        img = img.cuda()
        img = torch.cat((img, img, img), 1)  # MNIST通道1 -> 3
        source_label_list.append(label)
        source_img_list.append(img)

    source_img_list = torch.stack(source_img_list)
    source_img_list = source_img_list.view(-1, 3, 28, 28)

    # 获取目标测试样本
    target_label_list = []
    target_img_list = []
    for i, test_data in enumerate(target_test_loader):
        if i >= 16:
            break
        img, label = test_data
        label = label.numpy()
        img = img.cuda()
        target_label_list.append(label)
        target_img_list.append(img)

    target_img_list = torch.stack(target_img_list)
    target_img_list = target_img_list.view(-1, 3, 28, 28)

    # 合并源列表 + 目标列表
    combined_label_list = source_label_list
    combined_label_list.extend(target_label_list)
    combined_img_list = torch.cat((source_img_list, target_img_list), 0)

    source_domain_list = torch.zeros(512).type(torch.LongTensor)
    target_domain_list = torch.ones(512).type(torch.LongTensor)
    combined_domain_list = torch.cat((source_domain_list, target_domain_list), 0).cuda()

    print("Extract features to draw T-SNE plot...")
    combined_feature = combined_img_list  # combined_feature : 1024,3,28,28
    combined_feature = combined_feature.view(1024, -1)  # flatten

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    dann_tsne = tsne.fit_transform(combined_feature.detach().cpu().numpy())
    print('Draw plot ...')
    plot_embedding(dann_tsne, combined_label_list, combined_domain_list, 'input')

visualize_input()
