import torch
import torchvision
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import fetch_openml

datasets = fetch_openml("MNIST original",data_home='./dataset_mnist')
print(datasets)
def load_data():
    """transform to torch"""
    dataset_train = torchvision.datasets.MNIST(root='./dataset_mnist',
                                           train=True,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
    dataset_test = torchvision.datasets.MNIST(root='./dataset_mnist',
                                          train=False,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100, shuffle=True)  #600*100*([[28*28],x])
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=100, shuffle=False)
    return data_loader_train,data_loader_test

def show_images(dataloader):
    """图片形式查看数据集"""
    for i, (images,labels) in enumerate(dataloader):
        """绘制图片"""
        if (i+1)%100 == 0:
            print('{0}/{1}'.format(i,len(dataloader)))
            for j in range(len(images)):
                image = images[j].resize(28,28)
                plt.imshow(image)
                plt.axis('off')
                plt.title("batch_image")
                plt.show()

