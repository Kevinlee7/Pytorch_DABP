import torch
import mnist
import mnistm
import model
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import extract_features  # 假设你有一个提取特征的函数

# 定义保存名称
save_name = 'omg'

# 可视化函数
def visualize_tsne(features, title):
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3000)
    reduced_features = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.grid()
    plt.show()

def main():
    source_train_loader = mnist.mnist_train_loader
    target_train_loader = mnistm.mnistm_train_loader

    if torch.cuda.is_available():
        print('Running GPU : {}'.format(torch.cuda.current_device()))
        encoder = model.Extractor().cuda()
        classifier = model.Classifier().cuda()

        # 提取特征
        original_mnist_features = extract_features(encoder, classifier, source_train_loader)
        mnist_m_features = extract_features(encoder, classifier, target_train_loader)

        # 假设你已经从训练中获得了源域和 DANN 的特征
        source_only_features = extract_features(encoder, classifier, source_train_loader)  # 源域特征
        dann_features = extract_features(encoder, classifier, target_train_loader)  # DANN特征

        # 可视化每个数据集的特征
        visualize_tsne(original_mnist_features, "Original MNIST Features")
        visualize_tsne(mnist_m_features, "MNIST-M Features")
        visualize_tsne(source_only_features, "Features After Source Only Training")
        visualize_tsne(dann_features, "Features After DANN Training")

    else:
        print("There is no GPU -_-!")

if __name__ == "__main__":
    main()