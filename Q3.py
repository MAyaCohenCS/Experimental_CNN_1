import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

W = 32
H = 32
PATH = 'trained_models/cifar_net.pth'
PATH_default = 'trained_models/cifar_net.pth'
PATH_q3 = 'trained_models/cifar_net3.pth'
PATH_q4 = 'trained_models/cifar_net4.pth'


def get_permutation(w,h):
    '''
    get an index premutation for a h*w array
    :param w: widht of the array
    :param h: hight of the array
    :return: a tensor of indices indicating a permutation
    '''
    length = w * h
    r = torch.randperm(length)
    g = torch.add(r, length)
    b = torch.add(g, length)
    return torch.cat((r, g, b))


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    """Copy the neural network from the Neural Networks section before and modify
    it to take 3-channel images
    (instead of 1-channel images as it was defined)"""
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OurTransforms:
    '''
    a class that holds transforms that shuffle and reshiffle images
    '''
    def __init__(self, q_3 = False,q_4=False):
        self._shuffle = q_3 or q_4
        self.reshuffle = q_4
        self.idx_permute = None
        if self._shuffle:
            self.idx_permute = get_permutation(W, H)

    def reset_permutation(self):
        self.idx_permute = get_permutation(W, H)

    def shuffle_img(self, img):
        '''randomly reshuffle the spatial locations of pixels in the image'''
        if self.reshuffle: # Q4: no special structure
            self.reset_permutation()
        return img.view(-1)[self.idx_permute].view(3, W, H)

    def get_transform(self):
        '''get the relevent transform'''
        # shuffling receptive field
        if self._shuffle:
            return transforms.Compose(
        [transforms.ToTensor(),
         transforms.Lambda(self.shuffle_img),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #default:
        return transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def init_loaders(q_3=False, q_4=False):
    """
    Load and normalizing the CIFAR10 training and test datasets using torchvision
    :return: test loader and training loader
    """
    transform = OurTransforms(q_3, q_4).get_transform()
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def train_net(net, trainloader):
    #Define a Loss function and optimizer:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # Train the network
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            '''
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0'''
    print('Finished Training')
    torch.save(net.state_dict(), PATH)
    print(PATH)
    return net, inputs, labels


def display_some_images(loader):
    """
    showing some (4) images and printing their labels
    :param loader: the source from which images are loaded
    :return: the images displayed
    """
    # get some random images
    data_iter = iter(loader)
    images, labels = data_iter.next()
    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
    return images


def initially_test_network(test_loader):
    """
    display some random images and see how the network classifies them
    :param test_loader:
    """
    #  Let us display an image from the test set to get familiar:
    print("GroundTruth: ")
    images = display_some_images(test_loader)
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted:\n', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(4)))


def test_network(test_loader):
    """
    check out how the saved network performs
    :param test_loader:
    """
    net = Net()
    net.load_state_dict(torch.load(PATH))
    print("testing", PATH)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def test_class_performance(testloader):
    net = Net()
    net.load_state_dict(torch.load(PATH))
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            imges, lbls = data
            outputs = net(imges)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == lbls).squeeze()
            for i in range(4):
                label = lbls[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def test_locality(q_3=False, q_4=False):
    '''
    test how changes in localiy affect the performance of the NN
    :param q_3: if Q3 we will shuffle the images once
    :param q_4: if Q4 we will shuffle the images everytime its read
    :return: the image loaders
    '''
    # 1 Loading and normalizing CIFAR10:
    trainloader, testloader = init_loaders(q_3, q_4)
    global PATH
    if q_4:
        PATH = PATH_q4
    elif q_3:
        PATH = PATH_q3
    else:
        PATH = PATH_default
    # * Let us show some of the training images, for fun:
    display_some_images(trainloader)
    # 2 Define a Convolutional Neural Network:
    our_net = Net()
    # 3 Define a Loss function and optimizepr and 4 Train the network:
    train_net(our_net, trainloader)
    # 5 Test the network on the test data
    initially_test_network(testloader)
    # Let us look at how the network performs on the whole dataset:
    test_network(testloader)
    # what are the classes that performed well, and the classes that did not perform well?
    test_class_performance(testloader)
    return trainloader, testloader


if __name__ == '__main__':
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    test_locality()
    test_locality(q_3=True)
    test_locality(q_4=True)

