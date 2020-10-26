import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

data_x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 500)
data_y = np.sin(data_x)


def normalize_data(X, Y):
    """变换为 [n,1] 格式，符合torch格式"""
    X = np.expand_dims(X, axis=1)
    Y = Y.reshape(500, -1)
    return X, Y


def transform_dataloader(data_X, data_Y):
    """转化为pytorch的dataloader格式"""
    X, Y = normalize_data(data_X, data_Y)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float), torch.tensor(Y, dtype=torch.float))
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)
    return dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=10), nn.ReLU(),
            nn.Linear(10, 100), nn.ReLU(),
            nn.Linear(100, 10), nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, input: torch.FloatTensor):
        return self.net(input)


net = Net()


def train(dataloader):
    """训练"""
    data_X, data_Y = normalize_data(data_x, data_y)
    # 优化器 和 损失函数
    optim = torch.optim.Adam(net.parameters(net), lr=0.001)
    Loss = nn.MSELoss()

    for epoch in range(1000):
        loss = None
        for batch_x, batch_y in dataloader:
            y_predict = net(batch_x)
            loss = Loss(y_predict, batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()

        if (epoch + 1) % 100 == 0:
            print("step: {0} , loss: {1}".format(epoch + 1, loss.item()))

    print("训练完成......")


def plot_sin():
    test_x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 100)
    test_X = np.expand_dims(test_x, axis=1)

    plt.plot(data_x, data_y, label="fact")
    predict = net(torch.tensor(test_X, dtype=torch.float))
    plt.plot(test_X, predict.detach().numpy(), marker='x', label="predict")
    plt.title("fit_sin")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataloader = transform_dataloader(data_x, data_y)
    train(dataloader)
    plot_sin()
