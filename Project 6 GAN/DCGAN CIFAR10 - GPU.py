import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision import transforms
import torchvision
import sys                                  #for .py to accept argumements
import matplotlib.pyplot as plt
import time                                 #for calculating run time of training
import numpy as np
from google.colab import drive


def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def CIFAR10_data(batch_size_train):
    #------ Download Training and Testing Dataset.  * Mind the PATH here!
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.CIFAR10('./dataset/', train=False, download=True,    #download remarked False after downloaded once
      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),  transforms.Resize((64, 64)),
      torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])),            #0.1307, 0.3081 = Global mean, Global SD
      batch_size=batch_size_train, shuffle=False)
    return train_loader

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 4, stride = 2, padding = 1),
            #nn.Dropout2d(0.3),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size = 4, stride = 1, padding = 0),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self,x):
        #x = x.view(-1, 1, 28, 28)                       #it's fine for noise input, but real image needed to be reshaped
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

def images_to_vectors(images):
    return images.view(-1,784)

def vectors_to_images(vectors):
    return vectors.view(-1,1,28,28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        '''
        self.hidden0 = nn.Sequential(
            nn.Linear(100,256 * 7 * 7)
            #nn.LeakyReLU(0.2)
        )
        '''
        self.hidden0 = nn.Sequential(
            nn.ConvTranspose2d(100,512, kernel_size = 4, stride = 1, padding = 0),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.hidden1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.hidden2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size = 4, stride = 2, padding = 1),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        return x

def GenerateLatentVector(size, device):
    return torch.randn(size,100, 1, 1).to(device)    #produces a tensor with gaussian, zero mean, variance 1

def ones_target(size, device):
    return torch.ones(size,1).to(device)

def zeros_target(size, device):
    return torch.zeros(size,1).to(device)

def train_discriminator(optimizer,real_data,latent_data, discriminator, loss, device):
    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real,ones_target(real_data.size(0), device))  #Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities
    error_real.backward()

    prediction_fake = discriminator(latent_data)
    error_fake = loss(prediction_fake,zeros_target(real_data.size(0), device))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer,latent_data, discriminator, loss, device):
    optimizer.zero_grad()
    prediction = discriminator(latent_data)
    error = loss(prediction,ones_target(latent_data.size(0), device))
    error.backward()
    optimizer.step()
    return error

'''
def load_learning_state(generator, discriminator, d_optim, g_optim):
    generator_state_dict = torch.load('./results/generator.pth')
    discriminator_state_dict = torch.load('./results/discriminator.pth')
    d_optim_state_dict = torch.load('./results/d_optim.pth')
    g_optim_state_dict = torch.load('./results/g_optim.pth')
    return generator.load_state_dict(generator_state_dict), discriminator.load_state_dict(discriminator_state_dict), d_optim.load_state_dict(d_optim_state_dict), g_optim.load_state_dict(g_optim_state_dict)
'''
def save_learning_state(generator, discriminator, d_optim, g_optim):
    drive.mount('/content/gdrive')
    torch.save(generator.state_dict(), '/content/gdrive/generator.pth')         #save neural network state
    torch.save(discriminator.state_dict(), '/content/gdrive/discriminator.pth')         #save neural network state
    torch.save(d_optim.state_dict(), '/content/gdrive/d_optim.pth')   #save optimizer state
    torch.save(g_optim.state_dict(), '/content/gdrive/g_optim.pth')   #save optimizer state


def main(argv):

    torch.manual_seed(42)           #Generate 42 manual seeds and it's "reproducible" (fixed) for every run time
    torch.backends.cudnn.enabled = False    #Turn off CUDA, so all processing are serial to make sure result reproducible

    batch_size = 16
    data_loader = CIFAR10_data(batch_size)

    #---- plot original cifar10 datasets
    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


    # get some random training images
    dataiter = iter(data_loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    plt.show()
    #--- finished plotting cifar10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    generator = Generator()
    discriminator  = Discriminator()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    generator = generator.float()
    discriminator = discriminator.float()

    d_optim = optim.Adam(discriminator.parameters(),lr = 0.0002, betas = (0.5, 0.999) )
    g_optim = optim.Adam(generator.parameters(),lr = 0.0002, betas = (0.5, 0.999) )

    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = GenerateLatentVector(num_test_samples, device)
    test_noise = test_noise.to(device)

    #load_learning_state(generator, discriminator, d_optim, g_optim)

    print(generator(test_noise).shape)      #output : torch.Size([16, 3, 64, 64])

    num_epochs = 200

    d_error_list = []; g_error_list = []; epochs_list = [];
    start_time = time.time()


    for epoch in range(num_epochs):
        for n_batch, (real_batch, example_targets) in enumerate(data_loader):

            # train the discriminator
            #real_data = real_batch.view(-1,784)
            real_data = real_batch.to(device)
            latent_data = generator(torch.randn(batch_size,100, 1, 1, device=device)).detach()

            d_error,d_pred_real,d_pred_fake = train_discriminator(d_optim,real_data,latent_data, discriminator, loss, device)

            # train the generator
            latent_data = generator(GenerateLatentVector(batch_size, device))
            g_error = train_generator(g_optim,latent_data, discriminator, loss, device)

        d_error_list.append(d_error.item())
        epochs_list.append(epoch)
        g_error_list.append(g_error.item())

        if epoch % 20 == 0:
            print("Epoch: {}/{}".format(epoch, num_epochs))
            print("d_error (Error_real + error fake Loss)(Discriminator Loss): {}, Generator Loss: {}".format(d_error, g_error))
            #compare mean of real and fake tensor.  their value should come closer and closer
            print("D(x): {}, D(G(z)): {}".format(d_pred_real.mean(), d_pred_fake.mean()))
    

    save_learning_state(generator, discriminator, d_optim, g_optim)

    print("Run time (in second)", time.time() - start_time)

    test_images = generator(test_noise)
    
    # ------ show generated images 
    fig = plt.figure()
    title = "Epoch " + str(num_epochs)
    chart_global_title = fig.suptitle(title)
   
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        #plt.imshow(test_images[i][0], cmap='gray', interpolation = 'none')
        #plt.imshow(test_images[i][0].detach().numpy())
        #imshow(torchvision.utils.make_grid(test_images[i][0]))
        plt.axis("off")
        image = test_images[i].cpu()
        plt.imshow(np.transpose(image.detach().numpy(), (1,2,0)))
        #plt.title("Latent Vector example")
        plt.xticks([])          #remove xtick
        plt.yticks([])          #remove ytick
    plt.show()
     # ------ show generated images 

    #------ plot discriminator and adversarial loss over epoches
    fig = plt.figure()                                                                      #Plot of training and testing error
    plt.plot(epochs_list, d_error_list, color='blue')
    plt.plot(epochs_list, g_error_list, color='red')
    plt.legend(['Discriminator Loss', 'Adversarial Loss'], loc='upper right')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.show()
     #------ plot discriminator and adversarial loss over epoches

    return

if __name__ == "__main__":
    main(sys.argv)
