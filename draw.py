from __future__ import division
import  numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

def plotCM(matrix, savname,classes=None):
    """classes: a list of class names"""

    if(classes is None):
        classes=[]
        for i in range(1,matrix.shape[0]+1):
            classes.append("class_"+str(i))
    # Normalize by row
    matrix = matrix.astype(np.float)
    linesum = matrix.sum(1)
    linesum = np.dot(linesum.reshape(-1, 1), np.ones((1, matrix.shape[1])))
    matrix /= linesum
    # plot
    plt.switch_backend('agg')
    fig = plt.figure(figsize=(matrix.shape[0]+5,matrix.shape[0]))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str('%.2f' % (matrix[i, j] * 100)), va='center', ha='center')

    # ax.set_xticklabels([''] + classes, rotation=90)
    # ax.set_yticklabels([''] + classes)
    #save
    plt.savefig(savname)

def plotTSNE(features,label,savname="tsne.jpg"):
    tsne =TSNE(n_components=2, init='pca', random_state=501)
    X_tsne=tsne.fit_transform(features)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(label[i]), color=plt.cm.Set1(label[i]%8),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.savefig(savname)
