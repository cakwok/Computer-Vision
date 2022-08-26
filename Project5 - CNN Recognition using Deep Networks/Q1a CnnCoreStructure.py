'''
Wing Man Casca, Kwok
CS5330 Project 5 - Recognition using Deep Networks
'''
#import statements
import torch
import torchvision                   #Load Dataloader, 1. transformation eg cropping/normalization 2. GPU functions
import sys
import matplotlib.pyplot as plt      #Plot Graph
import torch.nn as nn                #Question 1C, Build a network
import torch.nn.functional as F      #Question 1C, Build a network
import torch.optim as optim          #Question 1C, Build a network

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
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            #print ("Ground Truth", target)
            #print ("log_softmax(x)", output)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

# main function
def main(argv):
    #------ Question 1A Get the MNIST digit data set
    n_epochs = 5                    #Epoch means run the network for 3 times(or num of loops here)
    batch_size_train = 64           #Num of training examples in 1 batch
    batch_size_test = 1000          #Num of testing examples in 1 batch
    learning_rate = 0.01            #How much to shift in each gradient descent
    momentum = 0.5
    log_interval = 10

    #random_seed = 1                #Generate 1 Random seed which changes every time
    torch.manual_seed(42)           #Generate 42 manual seeds and it's "reproducible" (fixed) for every run time, Question 1B
    torch.backends.cudnn.enabled = False    #Turn off CUDA, so all processing are serial to make sure result reproducible, Question 1B

    #------ Download Training and Testing Dataset.  * Mind the PATH here!
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./files/', train=False, download=False,    #download remarked False after downloaded once
      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.1307,), (0.3081,))])),            #0.1307, 0.3081 = Global mean, Global SD
      batch_size=batch_size_train, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./files/', train=False, download=False,      #download remarked False after downloaded once
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=False)
    #---------------------------------------------

    #enumerate is for loop like, but it returns indexes
    #the first enumerate statement defines the iteration
    examples = enumerate(test_loader)

    #next - read the first enumerated element
    batch_idx, (example_data, example_targets) = next(examples)

    #See one test data batch consists of torch.Size([1000, 1, 28, 28])
    #1000 examples of 28x28 pixels in grayscale (i.e. no rgb channels, hence the one)
    print (example_data.shape)

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation = 'none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])          #remove xtick
        plt.yticks([])          #remove ytick
    plt.show()

    NeuralNetwork1 = NeuralNetwork()                                                         #Initialize Neural Networks
    optimizer = optim.SGD(NeuralNetwork1.parameters(), lr=learning_rate, momentum=momentum)        #Initialize optimizer

    train_losses = []                                                                        #Question 1D Train the model
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    test(NeuralNetwork1, test_loader, test_losses)                                           #Run test once to see accuracy

    for epoch in range(1, n_epochs + 1):                                                     #Run training of network
        train(epoch, NeuralNetwork1, train_loader, optimizer, log_interval, train_losses, train_counter)
        test(NeuralNetwork1, test_loader, test_losses)


    fig = plt.figure()                                                                      #Plot of training and testing error
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
