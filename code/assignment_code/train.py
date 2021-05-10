'''
@File  :train.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 6:23 PM
@Desc  :train model [Batch size, channel, width, height]
'''

from model import Model
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from create_dataset import MyDataset

learning_rate = 1e-6
batch_size = 8
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)

train_data = MyDataset('../../data/12kDriveEnd_img/test.csv', '../../data/12kDriveEnd_img/train')
train_dataloader = DataLoader(train_data, batch_size=10, shuffle=True, drop_last=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (x, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = x.float().to(device)
        pred = model(X)
        loss = loss_fn(pred, y.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
print('done')
