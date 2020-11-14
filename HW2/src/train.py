import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import os


def calc_distribution():
    train_data = torchvision.datasets.CIFAR10('./data', train=True, download=True)
    # use np.concatenate to stick all the images together to form a 1600000 X 32 X 3 array
    x = np.concatenate([np.asarray(train_data[i][0]) for i in range(len(train_data))])
    print(x.shape)
    train_mean = np.mean(x, axis=(0, 1))
    train_std = np.std(x, axis=(0, 1))
    print(train_mean / 255, train_std / 255)


def save_checkpoint(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir / 'checkpoint.pth'
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir / 'best_model.pth'
        shutil.copyfile(f_path, best_fpath)


def plot_losses(train_losses, valid_losses, dropout, num_epochs, batch_size):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.plot(valid_losses, color='red', label='Validation loss')
    ax.set(title="Loss over epochs, using dropout =" + str(dropout) + ",epoch = " + str(num_epochs) + ",batch_size = "
                                                                                                      "" + str(
        batch_size),
           xlabel='Epoch',
           ylabel='Cross Entropy Loss')
    ax.legend()
    plt.tight_layout()
    fig.savefig('./Train_Valid_Results/loss.png', dpi=300)
    # fig.show()

    # change the plot style to default
    plt.style.use('default')


def plot_acc(train_acc_list, valid_acc_list, dropout, num_epochs, batch_size):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_acc_list)
    valid_losses = np.array(valid_acc_list)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training accuracy')
    ax.plot(valid_losses, color='red', label='Validation accuracy')
    ax.set(title="accuracy over epochs, using dropout ="
                 + str(dropout) + ",epoch = " + str(num_epochs) + ",batch_size = " + str(batch_size),
           xlabel='Epoch',
           ylabel='accuracy(%)')
    ax.legend()
    plt.tight_layout()
    fig.savefig('./Train_Valid_Results/acc.png', dpi=300)
    # fig.show()

    # change the plot style to default
    plt.style.use('default')


class ConvNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1, padding=2),  # in_channel = 3 , out_channel = 6, filter_size = 5
            nn.BatchNorm2d(6),  # 32*32*6, out_channel = 6(#feature)
            nn.ReLU(),  # activation of 1st layer=ReLu
            nn.Dropout2d(0.1),
            nn.MaxPool2d(2, stride=2))  # out : 32*32*6 -> 16*16*6
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=1, padding=2),  # in_channel = 6 , out_channel = 16, filter_size = 5
            nn.BatchNorm2d(16),  # 16*16*16
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))  # out : 16*16*16-> 8*8*16 = 1024
        self.layer3 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, 120),
            nn.BatchNorm1d(120),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU())
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(84, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.reshape(x.size(0), -1)  # (batch_size , -1)
        x = self.layer3(x)
        x = self.layer4(x)
        logits = self.output(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    correct_pred = 0
    n = 0

    for X, y_true in train_loader:
        optimizer.zero_grad()

        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)


        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()
        epoch_accuracy = 100*correct_pred.float() / n

        # Backward pass
        loss.backward()
        optimizer.step()

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, optimizer, epoch_loss, epoch_accuracy


def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''

    model.eval()
    running_loss = 0
    correct_pred = 0
    n = 0

    for X, y_true in valid_loader:
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass
        y_hat, y_prob = model(X)
        _, predicted_labels = torch.max(y_prob, 1)

        loss = criterion(y_hat, y_true)
        running_loss += loss.item() * X.size(0)
        n += y_true.size(0)
        correct_pred += (predicted_labels == y_true).sum()
        epoch_accuracy = 100*correct_pred.float() / n

    epoch_loss = running_loss / len(train_loader.dataset)

    return model, epoch_loss, epoch_accuracy


def training_loop(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device, print_interval=1):
    '''
    Function defining the entire training loop
    '''
    f = open("./Train_Valid_Results/log.txt", 'a')
    f.write("dropout =" + str(dropout) + ",epoch = " + str(num_epochs) + ",batch_size = " + str(batch_size))
    f.close()


    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    train_acc_list = []
    valid_acc_list = []

    # Train model
    for epoch in range(0, num_epochs):
        # training
        model, optimizer, train_loss, train_acc = train(train_loader, model, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_acc_list.append(train_acc)

        # validation
        with torch.no_grad():
            model, valid_loss,valid_acc = validate(valid_loader, model, criterion, device)
            valid_losses.append(valid_loss)
            valid_acc_list.append(valid_acc)

        if epoch % print_interval == (print_interval - 1):
            torch.save(model.state_dict(), './Train_Valid_Results/model.pth')
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch + 1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy(%): {train_acc:.2f}\t'
                  f'Valid accuracy(%): {valid_acc:.2f}')

            f = open("./Train_Valid_Results/log.txt", 'a')
            f.write(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch + 1}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy(%): {train_acc:.2f}\t'
                  f'Valid accuracy(%): {valid_acc:.2f}\n')
            f.close()

    plot_losses(train_losses, valid_losses, dropout, num_epochs, batch_size)
    plot_acc(train_acc_list, valid_acc_list, dropout, num_epochs, batch_size)

    return model, optimizer, (train_losses, valid_losses, train_acc_list, valid_acc_list)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    my_path = os.getcwd()

    num_classes = 10
    batch_size = 128

    learning_rate = 0.001

    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                          ])

    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                                         ])
    # CIFAR10 training set, test set 불러오기

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform_test)

    num_train = len(trainset)
    indices = list(range(num_train))
    train_idx, valid_idx = indices[10000:], indices[:10000]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=2)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    valid_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=valid_sampler,
                                               num_workers=2)

    num_epochs = 30

    dropout = 0.2

    calc_distribution()

    model = ConvNet(num_classes, dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model, optimizer, _ = training_loop(model, criterion, optimizer, train_loader, valid_loader, num_epochs, device)
