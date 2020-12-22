import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
import joblib

# 通过openml 读取 mnist数据集，较慢...
# mnist = fetch_openml('mnist_784', version=1, cache=True)
# X, y = mnist['data'], mnist['target']
# print(X.shape)
# print(y.shape)
# some_data, some_target = mnist['data'][10], mnist['target'][10]
# some_data_iamge = some_data.reshape(28, 28)
# plt.imshow(some_data_iamge, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("on")
# plt.savefig("some_digit_plot.png")


# 用sklearn自带的数据集 load_digits
digits = load_digits()
data = digits['data']
target = digits['target']
target = target.reshape(-1, 1)


# print(dir(digits))

predictor = svm.SVC(gamma='scale', C=1.0,
                    decision_function_shape='ovr', kernel='rbf')
predictor.fit(data[100:], target[100:])
joblib.dump(predictor, 'svm.pkl')
predictor = joblib.load('svm.pkl')


result = predictor.predict(digits.data[0:25])

for i in range(1,26):
    plt.subplot(5,5,i)
    plt.imshow(digits.images[i-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

print("实际值",digits.target[0:25])
print("预测值",result)
print("模型得分",predictor.score(data, target))






def load_data():
    """transform to torch"""
    dataset_train = torchvision.datasets.MNIST(root='./dataset',
                                               train=True,
                                               transform=torchvision.transforms.ToTensor(),
                                               download=True)
    dataset_test = torchvision.datasets.MNIST(root='./dataset',
                                              train=False,
                                              transform=torchvision.transforms.ToTensor(),
                                              download=True)

    data_loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=100,
                                                    shuffle=True)  # 600*100*([[28*28],x])
    data_loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=100, shuffle=False)
    return data_loader_train, data_loader_test


def show_images(dataloader):
    """图片形式查看数据集"""
    for i, (images, labels) in enumerate(dataloader):
        """绘制图片"""
        if (i + 1) % 100 == 0:
            print('{0}/{1}'.format(i, len(dataloader)))
            for j in range(len(images)):
                image = images[j].resize(28, 28)
                plt.imshow(image)
                plt.axis('off')
                plt.title("batch_image")
                plt.show()
