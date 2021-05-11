'''
@File  :train.py
@Author:Zhiwei Zheng
@Date  :5/9/2021 6:23 PM
@Desc  :train model [Batch size, channel, width, height]
'''

from model import Model
import torch
from torch import nn
from torch.utils.data import DataLoader
from create_dataset import MyDataset
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.optim.lr_scheduler import StepLR


def main():
    writer = SummaryWriter('./loss_log')
    learning_rate = 1e-3
    batch_size = 64
    epochs = 800

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Model().to(device)
    # for m in model.modules():
    #     if isinstance(m, (nn.Conv2d, nn.Linear)):
    #         nn.init.xavier_uniform_(m.weight)

    train_data = MyDataset('../../data/12kDriveEnd_img/train.csv', '../../data/12kDriveEnd_img/train')
    eval_data = MyDataset('../../data/12kDriveEnd_img/val.csv', '../../data/12kDriveEnd_img/val')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=True)


    loss_fn = nn.CrossEntropyLoss()
    base_optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(base_optimizer, T_max=40, eta_min=5 * 1e-6)
    scheduler_1 = ExponentialLR(base_optimizer, gamma=0.95)
    scheduler_2 = StepLR(base_optimizer, step_size=3, gamma=0.9)

    def test(epoch, dataloader, test_model, highest_score):
        size = len(dataloader.dataset)
        test_model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.float().to(device), y.to(device)
                pred = test_model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.int).sum().item()
        test_loss /= size
        correct /= size
        highest_score = max(highest_score, correct)
        print(f"\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        niter = epoch * len(train_dataloader)
        writer.add_scalars('Train',
                           {'val_correct': correct}, niter)
        return highest_score

    def train_loop(epoch, dataloader, train_model, loss_function, optimizer):
        print('learning_rate: ', optimizer.param_groups[0]['lr'])
        size = len(dataloader.dataset)
        for batch, (x, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = x.float().to(device)
            pred = train_model(X)
            loss = loss_function(pred, y.to(device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 5 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                niter = epoch * len(train_dataloader) + batch
                writer.add_scalars('Train',
                                   {'train_loss': loss}, niter)
                writer.add_scalars('Train',
                                   {'learning_rate': optimizer.param_groups[0]['lr']}, niter)

    for t in range(epochs):
        best_correct = 0
        before_correct = 0
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(t, train_dataloader, model, loss_fn, base_optimizer)
        best_correct = test(t, eval_dataloader, model, best_correct)
        scheduler_1.step()
        if (best_correct > 0.95 and best_correct > before_correct):
            torch.save(model, './model_'+str(best_correct)+'.pth')
        before_correct = before_correct
    print('done, best_score: ', best_correct)


if __name__ == '__main__':
    main()
