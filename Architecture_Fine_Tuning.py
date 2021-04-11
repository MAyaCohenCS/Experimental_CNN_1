import math

import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import utils.graphics as grph

matplotlib.use('TkAgg')  # )

# Data management
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=True, num_workers=1)


# Define net architecture.
# convolutions kernel size and first FC layer size as parameters
class DynamicNet(nn.Module):
    INPUT_SIZE = 32

    def __init__(self, conv_layer_size=5, fc_layer_size=120):
        super(DynamicNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, conv_layer_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, conv_layer_size)

        # calc dim of second conv output
        self.last_lay_size = DynamicNet.INPUT_SIZE
        self.last_lay_size = math.floor(
            (self.last_lay_size - (conv_layer_size - 1)) / 2)
        self.last_lay_size = math.floor(
            (self.last_lay_size - (conv_layer_size - 1)) / 2)
        assert self.last_lay_size == int(self.last_lay_size)
        self.last_lay_size = int(self.last_lay_size)

        self.fc1 = nn.Linear(16 * self.last_lay_size ** 2, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.last_lay_size ** 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(model, perf_logger):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):  # loop over the dataset multiple times
        train_loss = 0.0
        test_loss = 0.0

        # train over a single batch
        for i, train_batch in enumerate(trainloader, 0):
            inputs, labels = train_batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # collect statistics
            train_loss += loss.item()

            inputs, labels = next(iter(testloader))
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            if i % perf_logger.sampling_rate == perf_logger.sampling_rate - 1:
                avg_train_loss = train_loss / perf_logger.sampling_rate
                avg_test_loss = test_loss / perf_logger.sampling_rate

                perf_logger.read_data_point(avg_train_loss, avg_test_loss)
                print('[%d, %5d] train loss: %.3f, test loss: %.3f' %
                      (epoch + 1, i + 1, avg_train_loss, avg_test_loss))
                train_loss = 0.0
                test_loss = 0.0
    print('Finished Training')


kernel_sizes = [3, 5, 7, 9]
FC1_sizes = [15, 30, 70, 100, 200]

# start conv fine tuning
graph_title = 'Test and train loss across training procces, \naccording to kernel size in net architecture '
conv_pref = grph.Perf(desc=graph_title)  # initialize performance logger
for conv_ker_size in kernel_sizes:
    net = DynamicNet(conv_layer_size=conv_ker_size)  # initialize net
    conv_pref.new_session('ker edge: '+str(conv_ker_size))  # initialize data series on performance logger
    train(net, conv_pref)
conv_pref.plot_performance('conv_fine_tuning', show=False)

# start FC fine tuning
graph_title = 'Test and train loss across training procces, \naccording to first FC size in net architecture '
fc_pref = grph.Perf(desc=graph_title)  # initialize performance logger
for fc_size in FC1_sizes:
    net = DynamicNet(fc_layer_size=fc_size)
    conv_pref.new_session('FC1 size edge: '+str(fc_size))
    train(net, fc_pref)
conv_pref.plot_performance('fc_fine_tuning', show=False)
