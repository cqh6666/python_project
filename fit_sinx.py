

from torch.utils.data import DataLoader,TensorDataset
import torch.nn as nn
import numpy as np
import torch


x = np.linspace(-2.0 * np.pi, 2.0 * np.pi, 500)
np.random.shuffle(x)
y = np.sin(x)

x = np.expand_dims(x,axis=1)


y = y.reshape(500,-1)

dataset = TensorDataset(torch.tensor(x,dtype=torch.float),torch.tensor(y,dtype=torch.float))
dataloader = DataLoader(dataset,batch_size=100,shuffle=True)


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

