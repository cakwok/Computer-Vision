import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import os           #for importing files by all filenames
from PIL import Image
from Q1a_CnnCoreStructure import *
import Q1a_CnnCoreStructure
import csv
from torch.nn.functional import normalize #to normalize a tensor
import torchvision.transforms as transforms
from PIL import Image
import torchvision


#Question 3A Create a greek symbol dataset
def ConvertGreekImages(folder):

    f1 = open(folder + '/GreekPixels.csv', 'w')
    f1.write("Greek Letter Intensity \n")

    f2 = open(folder + '/GreekLabel.csv', 'w')
    f2.write('Greek Label \n')

    GroundTruth = []
    convert_tensor = transforms.ToTensor()
    HandwrittingImages_tensor = []
    HandwrittingImages = []

    for filename in os.listdir(folder):

        if filename[-3:] == "jpg":

            img = cv.imread(os.path.join(folder, filename), 0) #!! using imread has caused image becomes 3 channels!!  0 represents read as grayscale

            if img is not None:

                ret, img_T = cv.threshold(img, 180, 255, cv.THRESH_BINARY_INV)
                img_T = cv.resize(img_T, (28, 28))
                img_T = cv.cvtColor(img_T, cv.COLOR_BGR2RGB)

                print (filename)
                plt.imshow(img_T)
                plt.show()

                for r in range(28):
                    for c in range(28):
                        pixel = str(img_T[r][c][0]) + ", "      #third [] indicates channel.  in grayscale, 3 channels show the same value
                        f1.write(pixel)
                f1.write("\n")

                if filename.split("_")[0] == "alpha":
                    f2.write("0" + "\n")
                elif  filename.split("_")[0] == "beta":
                    f2.write("1" + "\n")
                elif  filename.split("_")[0] == "gamma":
                    f2.write("2" + "\n")

                cv.imwrite(folder + "/GreekFormatted/" + filename[:-3] + "png", cv.cvtColor(img_T, cv.COLOR_BGR2GRAY))

    f1.close()
    f2.close()

    return

class NeuralNetwork_FirstLayer(Q1a_CnnCoreStructure.NeuralNetwork):
    def __init__(self):
         NeuralNetwork.__init__(self)

    def forward(self, x):
        x = F.relu( F.max_pool2d( self.conv1(x), 2 ) )
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)) #A max pooling of dropout layer with a 2x2 window and a ReLU function applied
        x = x.view(-1, 320)                             #Fatten a tensor, since the input channel of previous layer is 320.
        x = F.relu(self.fc1(x))
        print ("Submodel(Q1a_CnnCoreStructure.NeuralNetwork)")
        return x

def trucate(network, test_loader):
    network_output = []
    network.eval()
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            network_output.append(output)
    return network_output[0]

def load_learnt_network_status(network, learning_rate, momentum):

    continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate,  momentum=momentum)

    #load the internal state of the network and optimizer when we last saved them.
    network_state_dict = torch.load('./results/model.pth')
    network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('./results/optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)

    return network, continued_optimizer

def loadGreekSymbols(folder):

    GreekSymbolPixel = []
    GreekSymbolLabel = []

    with open(folder + 'GreekPixels.csv') as f1:
        header = next(f1)
        reader = csv.reader(f1)                 #need to read as csv, otherwise it's typed as a string
        GreekSymbolPixel = list(reader)

    with open(folder + 'GreekLabel.csv') as f1:
        header = next(f1)
        for line in f1:
            GreekSymbolLabel.append(line.strip())   #.strip()!  otherwise "\n" would be included in the list

    return GreekSymbolPixel, GreekSymbolLabel

def convert1Dlist2tensor(GreekSymbolPixel):

    GreekSymbolPixel_tensor = []

    for i in GreekSymbolPixel:
        tensor = torch.FloatTensor(list(map(int, i))).resize(1, 28,28)       #convert a list into tensor, convert a list of strings into list of integers
        norm = normalize(tensor, p=3.0)                                      #normalize a tensor
        GreekSymbolPixel_tensor.append(norm)
        #GreekSymbolPixel_tensor.append(torch.FloatTensor(list(map(int, i))).resize(1, 28,28))
    return GreekSymbolPixel_tensor

def ProjectGreekSymbols(network, HandwrittingImages_tensor, GroundTruth):

    network.eval()
    pred_list = []
    elementVector = []

    with torch.no_grad():
        for data in HandwrittingImages_tensor:
            output = network(data)
            #pred = output.data.max(1, keepdim=True)[1]
            #pred_list.append(pred)
            elementVector.append(output)

    return elementVector

# ---- Question 3D compute distances in the embedding space
def compute_ssd(symbol, elementVector, testitemDescription):

    alpha_elementVector = []
    beta_elementVector = []
    gamma_elementVector = []
    alpha_ssd = []
    beta_ssd = []
    gamma_ssd = []

    print ("SSD between same classification" + testitemDescription)

    #Compute SSD between same letter
    for index, labels in enumerate(symbol):

        if labels == "0":
            if len(alpha_elementVector) == 0:
                alpha_elementVector = elementVector[index]

            ssd = np.sum((np.array(alpha_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
            print (ssd)
            alpha_ssd.append(ssd)

        elif labels == "1":
            if len(beta_elementVector) == 0:
                beta_elementVector = elementVector[index]

            ssd = np.sum((np.array(beta_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
            print (ssd)
            beta_ssd.append(ssd)

        elif labels == "2":
            if len(gamma_elementVector) == 0:
                gamma_elementVector = elementVector[index]

            ssd = np.sum((np.array(gamma_elementVector, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
            print (ssd)
            gamma_ssd.append(ssd)

    return alpha_elementVector, beta_elementVector, gamma_elementVector, alpha_ssd, beta_ssd, gamma_ssd

def print_ssd(GreekSymbolLabel, symbol, elementVector, label):

    ssd_list = []

    for index, labels in enumerate(GreekSymbolLabel):
        ssd = np.sum((np.array(symbol, dtype=np.float32) - np.array(elementVector[index], dtype=np.float32))**2)
        ssd_list.append([GreekSymbolLabel[index], ssd])         #create a list of list

    ssd_list_sorted = sorted(ssd_list,key=lambda l:l[0], reverse=False) #sort 2D array

    print(label + " SSD with other symbols")

    for items in ssd_list_sorted:
        print(items[0], items[1])

    print("Prediction:", min(ssd_list_sorted, key=lambda x:x[1]), "\n") #find the min ssd by searching list of list

    return min(ssd_list_sorted, key=lambda x:x[1])

def testMyHandwritting(network, HandwrittingImages_tensor, GroundTruth):                                          #Question 1G, test my handwritting
    network.eval()
    pred_list = []
    i = 0
    with torch.no_grad():
        for data in HandwrittingImages_tensor:
            output = network(data)
            pred = output.data.max(1, keepdim=True)[1]
            pred_list.append(pred)
    return pred_list

def main(argv):

    #ConvertGreekImages('greek-1')      #Dont remove!!!!!!!!!!  Convert Professor's dataset

    batch_size_train = 64           #Num of training examples in 1 batch
    batch_size_test = 1000          #Num of testing examples in 1 batch
    learning_rate = 0.01            #How much to shift in each gradient descent
    momentum = 0.5

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/', train=False, download=False,      #download remarked False after downloaded once
    transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=False)

    showNNFirstDenseLayer = NeuralNetwork_FirstLayer()
    print(showNNFirstDenseLayer)

    showNNFirstDenseLayer, continued_optimizer = load_learnt_network_status(showNNFirstDenseLayer, learning_rate, momentum)

    showNNFirstDenseLayer_truncated_output = trucate(showNNFirstDenseLayer, test_loader)
    print("showNNFirstDenseLayer_truncated_output")
    print(showNNFirstDenseLayer_truncated_output.size())

    # ---- Question 1C, project geek symbols into the embedding space
    # ---- read csv
    GreekSymbolPixel, GreekSymbolLabel = loadGreekSymbols("./")

    # ---- convert 1D list into tensor format
    GreekSymbolPixel_tensor = convert1Dlist2tensor(GreekSymbolPixel)

    #print (GreekSymbolPixel_tensor[0])
    print (GreekSymbolPixel_tensor[0].size())

    # ---- Get a set of 27 x 50 element vectors
    elementVector = ProjectGreekSymbols(showNNFirstDenseLayer, GreekSymbolPixel_tensor, GreekSymbolLabel)

    #print ("elementVector")
    #print (elementVector[0])
    print (len(elementVector), elementVector[0].size())

    # ---- Question 3D compute distances in the embedding space
    alpha_elementVector, beta_elementVector, gamma_elementVector, alpha_ssd, beta_ssd, gamma_ssd = compute_ssd(GreekSymbolLabel, elementVector, "MNIST Handwritting")

    print("alpha_ssd")
    print(alpha_ssd)

    print("beta ssd")
    print(beta_ssd)

    print("gamma_ssd")
    print(gamma_ssd)

    #Compute SSD between different letter
    #--- print out ssd
    print_ssd(GreekSymbolLabel, alpha_elementVector, elementVector, "Alpha")
    print_ssd(GreekSymbolLabel, beta_elementVector, elementVector, "Beta")
    print_ssd(GreekSymbolLabel, gamma_elementVector, elementVector, "Gamma")

    #---- Question 3E, create your own greek symbol data
    #ConvertGreekImages('CascaHandWritting/Greek')       #Convert my own Greek handwritting
    GreekSymbolPixel_handwrite, GreekSymbolLabel_handwrite = loadGreekSymbols("CascaHandWritting/Greek/")
    #print ("GreekSymbolPixel_handwrite")
    #print (GreekSymbolPixel_handwrite)

    GreekSymbolPixel_tensor_handwrite = convert1Dlist2tensor(GreekSymbolPixel_handwrite)

    #print (GreekSymbolPixel_tensor_handwrite[0])
    print (GreekSymbolPixel_tensor_handwrite[0].size())

    # ---- Get a set of 27 x 50 element vectors, my handwritting
    elementVector_handwrite = ProjectGreekSymbols(showNNFirstDenseLayer, GreekSymbolPixel_tensor_handwrite, GreekSymbolLabel_handwrite)
    #print ("elementVector_handwrite")
    #print (elementVector_handwrite[0])
    print (len(elementVector_handwrite), elementVector_handwrite[0].size())

    #--- compute ssd
    alpha_elementVector_handwrite, beta_elementVector_handwrite, gamma_elementVector_handwrite, alpha_ssd_handwrite, beta_ssd_handwrite, gamma_ssd_handwrite = compute_ssd(GreekSymbolLabel_handwrite, elementVector_handwrite, "Greek Handwritting")

    print("alpha_ssd_handwrite")
    print(alpha_ssd_handwrite)

    print("beta ssd_handwrite")
    print(beta_ssd_handwrite)

    print("gamma_ssd_handwrite")
    print(gamma_ssd_handwrite)

    #--- print out ssd
    pred_list_handGreek = []
    pred_list_handGreek.append(print_ssd(GreekSymbolLabel, alpha_elementVector_handwrite, elementVector, "Alpha"))
    pred_list_handGreek.append(print_ssd(GreekSymbolLabel, beta_elementVector_handwrite, elementVector, "Beta"))
    pred_list_handGreek.append(print_ssd(GreekSymbolLabel, gamma_elementVector_handwrite, elementVector, "Gamma"))
    print (pred_list_handGreek)

    HandwrittingGreekImages = []
    HandwrittingGreekImages.append(cv.imread("CascaHandWritting/Greek/GreekFormatted/gamma_2_casca.png"))
    HandwrittingGreekImages.append(cv.imread("CascaHandWritting/Greek/GreekFormatted/beta_1_casca.png"))
    HandwrittingGreekImages.append(cv.imread("CascaHandWritting/Greek/GreekFormatted/alpha_1_casca.png"))

    fig = plt.figure()
    for i in range(3):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(HandwrittingGreekImages[i], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format((pred_list_handGreek[i][0])))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)

    
