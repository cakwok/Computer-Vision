import torch
import torchvision                   #Load Dataloader, 1. transformation eg cropping/normalization 2. GPU functions
import sys
import matplotlib.pyplot as plt      #Plot Graph
import torch.nn as nn                #Question 1F, Read the network and run it on the test set
import torch.nn.functional as F      #Question 1C, Build a network
import torch.optim as optim          #Question 1C, Build a network
import numpy as np
import cv2 as cv

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

class Submodel(NeuralNetwork):
    def __init__(self):
         NeuralNetwork.__init__(self)

    def forward(self, x):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) )
        x = F.relu( F.max_pool2d( self.conv2_drop( self.conv2(x)), 2 ) )
        return x

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
            #print ("Ground Truth", target)
            #print ("log_softmax(x)", output)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            network_output.append(output)
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return network_output[0]

def trucate(network, test_loader):
    network_output = []
    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            network_output.append(output)
    return network_output[0]

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
    print (example_data.shape)

    first_trainining_dataset = enumerate(train_loader)
    batch_idx, (first_training_data, first_training_targets) = next(first_trainining_dataset)

    # -- Continue training from the state_dicts we saved during our first training run

    #initialize a new set of network and optimizers.

    continued_network = NeuralNetwork()

    #----- Question 2A, print out weight and shape of the first layers
    print("continued_network")
    #print(continued_network.conv1)     #print out whole neural network structure
    print(continued_network)          #print out shape of conv1 layer only

    print("continued_network.conv1.weight")
    print(continued_network.conv1.weight)
    print("continued_network.conv1.shape")
    print(continued_network.conv1.weight.shape)

    #----- Question 2A, visualize the ten filters

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        #plt.imshow(continued_network.conv1.weight[i][0].detach().numpy(), cmap='gray', interpolation='none')        #plt.title("Prediction: {}".format(np.argmax(output[i]), continued_network.data.max(1, keepdim=True)[1][i].item()))
        plt.imshow(continued_network.conv1.weight[i][0].detach().numpy(), interpolation='none')                     #to plot it in green, remove cmap='gray'
        plt.title("Filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    #----- Question 2B, Show the effect of the filters

    fig = plt.figure()
    j = 1
    for i in range(10):
        plt.subplot(5,4,j)
        plt.tight_layout()
        plt.imshow(continued_network.conv1.weight[i][0].detach().numpy(),cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        j += 1
        plt.subplot(5,4,j)
        img_f = cv.filter2D(first_training_data[0][0].detach().numpy(), -1, continued_network.conv1.weight[i][0].detach().numpy())
        plt.imshow(img_f, cmap='gray', interpolation='none')
        j += 1
        plt.xticks([])
        plt.yticks([])
    plt.show()

    #----- Question 2c, build a truncated model

    NeuralNetwork_truncated = Submodel()
    print("NeuralNetwork_truncated")
    print(NeuralNetwork_truncated)

    continued_optimizer = optim.SGD(NeuralNetwork_truncated.parameters(), lr=learning_rate,  momentum=momentum)

    #load the internal state of the network and optimizer when we last saved them.
    network_state_dict = torch.load('./results/model.pth')
    NeuralNetwork_truncated.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('./results/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)

    NeuralNetwork_truncated_output = trucate(NeuralNetwork_truncated, test_loader)
    print("NeuralNetwork_truncated_output")
    print(NeuralNetwork_truncated_output.size())

    fig = plt.figure()
    j = 1
    for i in range(10):
        plt.subplot(5,4,j)
        plt.tight_layout()
        plt.imshow(NeuralNetwork_truncated.conv2.weight[i][0].detach().numpy(),cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        j += 1
        plt.subplot(5,4,j)
        img_f = cv.filter2D(first_training_data[0][0].detach().numpy(), -1, NeuralNetwork_truncated.conv2.weight[i][0].detach().numpy())
        plt.imshow(img_f, cmap='gray', interpolation='none')
        j += 1
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
