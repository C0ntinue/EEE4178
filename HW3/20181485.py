import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

# ================================================================== #
#                        0. Define Hyper-parameters
# ================================================================== #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
# num_epochs = 4 RNN type에 따라 다르게 설정하였다.
learning_rate = 0.001
dropout = 0

# rnn_type = 'RNN'  # select from 'RNN', 'LSTM' and 'GRU'.
rnn_types = ['RNN','LSTM','GRU']
# ================================================================== #
#                        0.1 Define utils
# ================================================================== #

student_id = '20181485'


def plot_losses(train_losses, dropout, num_epochs, batch_size,rnn_type):
    '''
    Function for plotting training and validation losses
    '''

    # temporarily change the style of the plots to seaborn
    plt.style.use('seaborn')

    train_losses = np.array(train_losses)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(train_losses, color='blue', label='Training loss')
    ax.set(title=rnn_type+"\'s Loss over 1 data sample, using dropout =" + str(dropout) + ",epoch = " + str(
        num_epochs) \
                        + ","
                                                                                                      "batch_size = "
                                                                                                      "" + str(
        batch_size),
           xlabel='number of data sample',
           ylabel='Cross Entropy Loss')
    ax.legend()
    plt.tight_layout()
    fig.savefig('./Train_Results/'+rnn_type+'_loss.png', dpi=300)
    # fig.show()

    # change the plot style to default
    plt.style.use('default')


# ================================================================== #
#                        1. Load Data
# ================================================================== #
train_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='./datasets/',
                                          train=False,
                                          transform=transforms.ToTensor())

# ================================================================== #
#                        2. Define Dataloader
# ================================================================== #
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# ================================================================== #
#                        3. Define Model
# ================================================================== #


class RNN(nn.Module):
    """
    Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
    """

    def __init__(self, rnn_type, input_size, hidden_size, num_layers, num_classes, dropout=0, bidirectional=False):
        super(RNN, self).__init__()
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_direction = int(bidirectional) + 1
        # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, 1, dropout=dropout, batch_first=True,
                                         bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x,None)  # out: tensor of shape (batch_size, seq_length, hidden_size),
        # None represents zero hidden/cell initial state!

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # [1, 128] [128 10] [1 10]
        return out




# ================================================================== #
#                        5. Train / Test
# ================================================================== #
if __name__ == '__main__':

    for rnn_type in rnn_types:
        model = RNN(rnn_type, input_size, hidden_size, num_layers, num_classes).to(device)

        if rnn_type == 'RNN':
            num_epochs = 30
        else:
            num_epochs = 10

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate * 0.01,
                                     amsgrad=True)

        print(rnn_type+':')
        total_step = len(train_loader)
        # set objects for storing metrics
        train_losses = []

        for epoch in range(num_epochs):
            for i, (images, labels) in enumerate(train_loader):

                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                train_losses.append(loss)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                    torch.save(model.state_dict(), './Train_Results/' + rnn_type + '_' + student_id + '.pth')

        # plot losses
        plot_losses(train_losses, dropout, num_epochs, batch_size, rnn_type)

        # print model's, optimizer's state dict
        # print("Model's state_dict:")
        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # print("Optimizer's state_dict:")
        # for var_name in optimizer.state_dict():
        #     print(var_name, "\t", optimizer.state_dict()[var_name])

        # load model
        test_model = RNN(rnn_type, input_size, hidden_size, num_layers, num_classes).to(device)
        checkpoint = torch.load('./Train_Results/' + rnn_type + '_' + student_id + '.pth', map_location=device)
        test_model.load_state_dict(checkpoint)

        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, sequence_length, input_size).to(device)
                labels = labels.to(device)
                outputs = test_model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

