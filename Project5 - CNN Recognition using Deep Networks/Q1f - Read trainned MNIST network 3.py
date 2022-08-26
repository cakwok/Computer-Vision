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

def train(epoch, network, train_loader, optimizer, log_interval, train_losses, train_counter):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
            #100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))

def test(network, test_loader, test_losses):                                                     #Question 1D, test the network
    network.eval()
    test_loss = 0
    correct = 0
    network_output = []
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            network_output.append(output)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return network_output[0]

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
            if pred == GroundTruth[i]:
                correct += 1
            i += 1
    print ("correctness", correct)
    print ("pred_list", pred_list)
    #print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return pred_list

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

def main(argv):
    n_epochs = 5                    #Epoch means run the network for 3 times(or num of loops here)
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

    #initialize a new set of network and optimizers.
    continued_network = Q1a_CnnCoreStructure.NeuralNetwork()
    continued_optimizer = optim.SGD(continued_network.parameters(), lr=learning_rate,  momentum=momentum)

    #load the internal state of the network and optimizer when we last saved them.
    network_state_dict = torch.load('./results/model.pth')
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('./results/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)
    #--------------------------------------------

    train_losses = []                                                                        #Question 1D Train the model
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    output = test(continued_network, test_loader, test_losses)

    print ("            Neural Network Output                                   Ground truth    Max Index")
    print ("---------------------------------------------------------------------------------------------")

    fig = plt.figure()
    for i in range(10):
        if i <9:
            plt.subplot(3,3,i+1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
            #plt.title("Prediction: {}".format(np.argmax(output[i]), continued_network.data.max(1, keepdim=True)[1][i].item()))
            plt.title("Prediction: {}".format(np.argmax(output[i])))
            plt.xticks([])
            plt.yticks([])

        print (str(list(np.around(np.array(output[i]), 2))) + "\t", str(example_targets[i].item()) + "\t", np.argmax(output[i]).item())
    plt.show()

    #---------------------- Question 1G, recognise handwritting
    GroundTruth, HandwrittingImages_tensor, HandwrittingImages = LoadHandwrittingImages("CascaHandWritting/formatted")

    predict_list = testMyHandwritting(continued_network, HandwrittingImages_tensor, GroundTruth)

    fig = plt.figure()
    for i in range(len(HandwrittingImages)):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(HandwrittingImages[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format((predict_list[i].item())))
        plt.xticks([])
        plt.yticks([])
    plt.show()



    return

if __name__ == "__main__":
    main(sys.argv)
