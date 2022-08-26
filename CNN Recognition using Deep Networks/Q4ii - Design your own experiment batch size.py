import torch
import torchvision                   #Load Dataloader, 1. transformation eg cropping/normalization 2. GPU functions
import sys
import matplotlib.pyplot as plt      #Plot Graph
import torch.nn as nn                #Question 1F, Read the network and run it on the test set
import torch.nn.functional as F      #Question 1C, Build a network
import torch.optim as optim          #Question 1C, Build a network
import Q1a_CnnCoreStructure
from Q1a_CnnCoreStructure import *
import numpy as np
import os                               #for importing files by all filenames
import cv2 as cv
from PIL import Image
import torchvision.transforms as transforms

class NeuralNetwork(nn.Module):               #Question 1C, Build a neural network
    def __init__(self):                       #Initialize all neural network layers
        super(NeuralNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)    #A convolution layer with 1 in channel, 10 out channels, 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)   #A convolution layer with 10 in channel, 20 out channels, 5x5 filters
        self.conv1_drop = nn.Dropout2d(p=0.5)                #Create a dropout layer with a 0.5 dropout rate (50%)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)                   # Creates fully connected Linear layer with 10 nodes and the log_softmax function applied to the output.
        self.fc2 = nn.Linear(50, 10)                   # #1 parameters: incoming signals, #2 parameter: output nodes/signal

    def forward(self, x):                               #Define forward pass per layer
        x = F.relu(F.max_pool2d(self.conv1(x), 2))      #A max pooling layer with a 2x2 window and a ReLU function applied
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #A max pooling of dropout layer with a 2x2 window and a ReLU function applied
        x = x.view(-1, 320)                             #Fatten a tensor, since the input channel of previous layer is 320.
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training) #??? still need???
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter):           #Question 1D, train the network
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):   #data = handwritting image, target = ground truth
        optimizer.zero_grad()                           #initialize optimizer to 0 each epoch.  Default accumulative
        output = network(data)                   #input handwritting image and run training
        loss = F.nll_loss(output, target)               #after training has run, compare output v/s ground truth
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
            #100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')         #save neural network state, Question 1E
            torch.save(optimizer.state_dict(), './results/optimizer.pth')   #save optimizer state, Question 1E

def test(network, test_loader, test_losses):                                                     #Question 1D, test the network
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():                   #no_grad means no gradient magnitude
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))


def testMyHandwritting(network, HandwrittingImages_tensor, GroundTruth):                                          #Question 1G, test my handwritting
    network.eval()
    test_loss = 0
    correct = 0
    pred_list = []
    i = 0
    with torch.no_grad():
        for data in HandwrittingImages_tensor:
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_list.append(pred)
            if str(pred.item()) == GroundTruth[i]:      #change tensor.item into a string to compare filename
                correct += 1
            i += 1
    print ("correctness", correct)
    print ("pred_list", pred_list)
    #print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct/10, pred_list

def LoadHandwrittingImages(folder):
    HandwrittingImages = []
    HandwrittingImages_tensor = []
    GroundTruth = []
    convert_tensor = transforms.ToTensor()
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename)) #!! using imread has caused image becomes 3 channels!!
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        GroundTruth.append(filename[0])
        if img is not None:
            tensor = convert_tensor(img)
            HandwrittingImages_tensor.append(tensor)
            print ("tensor.size", tensor.size())
            print ("filename", filename)
            HandwrittingImages.append(img)
    return GroundTruth, HandwrittingImages_tensor, HandwrittingImages

def GetEpochErrorList(NeuralNetworkX, learning_rate, momentum, test_loader, train_loader, log_interval, train_losses, train_counter, test_losses, n_epochs, HandwrittingImages_tensor, GroundTruth):

    optimizer = optim.SGD(NeuralNetworkX.parameters(), lr = learning_rate, momentum = momentum)

    #test(NeuralNetworkX, test_loader, test_losses)

    for epoch in range(1, n_epochs + 1):                                                     #Run training of network
        train(epoch, NeuralNetworkX, train_loader, optimizer, log_interval, train_losses, train_counter)
        #test(NeuralNetwork1, test_loader, test_losses)

    error_rate, predict_list = testMyHandwritting(NeuralNetworkX, HandwrittingImages_tensor, GroundTruth)

    return NeuralNetworkX, error_rate

def main(argv):
    n_epochs = 5                   #Epoch means run the network for 3 times(or num of loops here)
    batch_size_train = 64           #Num of training examples in 1 batch
    batch_size_test = 1000          #Num of testing examples in 1 batch
    learning_rate = 0.01            #How much to shift in each gradient descent
    momentum = 0.5
    log_interval = 10

    #random_seed = 1                #Generate 1 Random seed which changes every time
    torch.manual_seed(42)           #Generate 42 manual seeds and it's "reproducible" (fixed) for every run time, Question 1B
    torch.backends.cudnn.enabled = False

    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=False,    #download remarked False after downloaded once
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])),            #0.1307, 0.3081 = Global mean, Global SD
    batch_size=batch_size_train, shuffle=False)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=False,      #download remarked False after downloaded once
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=False)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print("example_data, example_targets")
    print(example_data, example_targets)
    print ("tensor size", example_data.size())

    # -- Continue training from the state_dicts we saved during our first training run
    NeuralNetwork1 = NeuralNetwork()                                                         #Initialize Neural Networks
    optimizer = optim.SGD(NeuralNetwork1.parameters(), lr=learning_rate, momentum=momentum)        #Initialize optimizer

    #--------------------------------------------
    train_losses = []                                                                        #Question 1D Train the model
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    test(NeuralNetwork1, test_loader, test_losses)

    # ---- prepare data to plot error rate of number of epoch
    epoch_list = []

    for epoch in range(1, n_epochs + 1):                                                     #Run training of network
        train(epoch, NeuralNetwork1, train_loader, optimizer, log_interval, train_losses, train_counter)
        test(NeuralNetwork1, test_loader, test_losses)

    epoch_list.append(epoch)

    #----------------------  recognise handwritting and get error rate
    error_list = []

    GroundTruth, HandwrittingImages_tensor, HandwrittingImages = LoadHandwrittingImages("CascaHandWritting/formatted")
    print ("GroundTruth")
    print (GroundTruth)

    print("batch_size_train", batch_size_train)
    error_rate, predict_list = testMyHandwritting(NeuralNetwork1, HandwrittingImages_tensor, GroundTruth)
    error_list.append(error_rate)

    #---- plot handwritting recognistion for the first batch size of training

    fig = plt.figure()
    for i in range(len(HandwrittingImages)):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(HandwrittingImages[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format((predict_list[i].item())))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    for i in range (128, 448, 64):

        batch_size_train =  i

        for epoch in range(1, n_epochs + 1):                                                     #Run training of network
            train(epoch, NeuralNetwork1, train_loader, optimizer, log_interval, train_losses, train_counter)

        print("batch_size_train", batch_size_train)

        error_rate, predict_list = testMyHandwritting(NeuralNetwork1, HandwrittingImages_tensor, GroundTruth)
        #error_list.append(error_rate)

        print("error_rate", error_rate)

    return

if __name__ == "__main__":
    main(sys.argv)
