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

# Define net architecture
class LinearNet(nn.Module):
    def __init__(self, fc1_layer_size=120):
        super(LinearNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 24 * 24, fc1_layer_size)
        self.fc2 = nn.Linear(fc1_layer_size, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 16 * 24 * 24)
        x = self.fc1(x)
        x = self.fc2(x)
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


# Net Training
# start conv fine tuning
graph_title = 'Test and train loss across training procces of a linear model, according to FC1 size'
pref = grph.Perf(desc=graph_title)  # initialize performance logger

pref.new_session('basic linear model \nFC1 =120')
linear_net = LinearNet()
train(linear_net, pref)
pref.plot_performance('linear_models', show=False)

pref.new_session('extended linear model \nFC1 =240')
linear_net_extended_fc = LinearNet(240)
train(linear_net_extended_fc, pref)

pref.plot_performance('linear_models', show=False)

