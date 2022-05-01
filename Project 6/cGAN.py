import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision import transforms
import torchvision
import sys                                  #for .py to accept argumements
import matplotlib.pyplot as plt
import time                                 #for calculating run time of training

def images_to_vectors(images):
    return images.view(images.size(0), 784)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

def mnist_data(batch_size_train):
    #------ Download Training and Testing Dataset.  * Mind the PATH here!
    train_loader = torch.utils.data.DataLoader(
      torchvision.datasets.MNIST('./dataset/', train=False, download=True,    #download remarked False after downloaded once
      transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (0.5,))])),            #0.1307, 0.3081 = Global mean, Global SD
      batch_size=batch_size_train, shuffle=False)

    return train_loader

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator,self).__init__()
        self.label_embedding = nn.Embedding(10, 10)

        self.hidden0 = nn.Sequential(
            nn.Linear(784 + 10, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256, 1),
             nn.Sigmoid()
        )


    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x, c], 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

    #dynamic flattening size here, because the input could be latent vector 1 x 100 or real image 28 x 28
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def images_to_vectors(images):
    return images.view(-1,784)

def vectors_to_images(vectors):
    return vectors.view(-1,1,28,28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()

        #This module is often used to store word embeddings and retrieve them using indices.
        #The input to the module is a list of indices, and the output is the corresponding word embeddings.
        self.label_embedding = nn.Embedding(10,10 )

        self.hidden0 = nn.Sequential(
            nn.Linear(100 + 10, 256),     #100 for latent vector, 10 for label
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x, labels):
        c = self.label_embedding(labels)
        x = torch.cat([x,c], 1)
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return x

def GenerateLatentVector(size):
    return torch.randn(size,100)    #produces a tensor with gaussian, zero mean, variance 1

def ones_target(size):
    return torch.ones(size,1)

def zeros_target(size):
    return torch.zeros(size,1)

def train_discriminator(optimizer,real_data,latent_data, discriminator, loss, example_targets, fake_labels):
    optimizer.zero_grad()

    prediction_real = discriminator(real_data, example_targets)
    error_real = loss(prediction_real,ones_target(real_data.size(0)))  #Creates a criterion that measures the Binary Cross Entropy between the target and the input probabilities
    error_real.backward()

    prediction_fake = discriminator(latent_data, fake_labels)
    error_fake = loss(prediction_fake,zeros_target(real_data.size(0)))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer,latent_data, discriminator, loss, fake_labels):
    optimizer.zero_grad()
    prediction = discriminator(latent_data, fake_labels)
    error = loss(prediction,ones_target(latent_data.size(0)))
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

def save_learning_state(generator, discriminator, d_optim, g_optim):
    torch.save(generator.state_dict(), './results/generator.pth')         #save neural network state
    torch.save(discriminator.state_dict(), './results/discriminator.pth')         #save neural network state
    torch.save(d_optim.state_dict(), './results/d_optim.pth')   #save optimizer state
    torch.save(g_optim.state_dict(), './results/g_optim.pth')   #save optimizer state
'''

def main(argv):

    torch.manual_seed(42)           #Generate 42 manual seeds and it's "reproducible" (fixed) for every run time
    torch.backends.cudnn.enabled = False    #Turn off CUDA, so all processing are serial to make sure result reproducible

    batch_size = 16
    data_loader = mnist_data(batch_size)

    generator = Generator()
    discriminator  = Discriminator()

    d_optim = optim.Adam(discriminator.parameters(),lr = 0.0002, betas = (0.5, 0.999) )
    g_optim = optim.Adam(generator.parameters(),lr = 0.0002, betas = (0.5, 0.999) )

    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = GenerateLatentVector(num_test_samples)

    #load_learning_state(generator, discriminator, d_optim, g_optim)

    num_epochs = 100

    d_error_list = []; g_error_list = []; epochs_list = [];
    start_time = time.time()

    for epoch in range(num_epochs):
        for n_batch, (real_batch, example_targets) in enumerate(data_loader):

            true_labels = torch.ones(batch_size)
            fake_labels = torch.randint(0, 10, (batch_size,))

            # train the discriminator
            #real_data = real_batch.view(-1,784)
            real_data = real_batch.view(batch_size, 784)
            latent_data = generator(torch.randn(batch_size,100), fake_labels).detach()


            d_error,d_pred_real,d_pred_fake = train_discriminator(d_optim,real_data,latent_data, discriminator, loss, example_targets, fake_labels)

            # train the generator
            latent_data = generator(GenerateLatentVector(batch_size), fake_labels)
            g_error = train_generator(g_optim,latent_data, discriminator, loss, fake_labels)

        d_error_list.append(d_error.item())
        epochs_list.append(epoch)
        g_error_list.append(g_error.item())

        if epoch % 20 == 0:
            print("Epoch: {}/{}".format(epoch, num_epochs))
            print("d_error (Error_real + error fake Loss)(Discriminator Loss): {}, Generator Loss: {}".format(d_error, g_error))
            #compare mean of real and fake tensor.  their value should come closer and closer
            print("D(x): {}, D(G(z)): {}".format(d_pred_real.mean(), d_pred_fake.mean()))

    #save_learning_state(generator, discriminator, d_optim, g_optim)

    #print(d_error_list)
    #print(g_error_list)

    print("Run time (in second)", time.time() - start_time)

    fake_labels = torch.randint(0, 10, (batch_size,))
    test_images = vectors_to_images(generator(test_noise, fake_labels))
    test_images = test_images.data


    fig = plt.figure()
    title = "Epoch " + str(num_epochs)
    chart_global_title = fig.suptitle(title)
    # show generated images
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(test_images[i][0], cmap='gray', interpolation = 'none')
        plt.title("Ground Truth: {}".format(fake_labels[i]))
        plt.xticks([])          #remove xtick
        plt.yticks([])          #remove ytick
    plt.show()

    #plot discriminator and adversarial loss over epoches
    fig = plt.figure()                                                                      #Plot of training and testing error
    plt.plot(epochs_list, d_error_list, color='blue')
    plt.plot(epochs_list, g_error_list, color='red')
    plt.legend(['Discriminator Loss', 'Adversarial Loss'], loc='upper right')
    plt.xlabel('Epoches')
    plt.ylabel('Loss')
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)
