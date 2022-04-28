import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torchvision import transforms
import torchvision
import sys
import matplotlib.pyplot as plt

'''
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, ), (.5, ))
        ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)# Load data
data = mnist_data()# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)
'''

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
        self.hidden0 = nn.Sequential(
            nn.Linear(784,1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def images_to_vectors(images):
    return images.view(-1,784)
def vectors_to_images(vectors):
    return vectors.view(-1,1,28,28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.hidden0 = nn.Sequential(
            nn.Linear(100,256),
            nn.LeakyReLU(0.2)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(256,512),
            nn.LeakyReLU(0.2)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(512,1024),
            nn.LeakyReLU(0.2)
        )
        self.out = nn.Sequential(
            nn.Linear(1024,784),
            nn.Tanh()
        )

    def forward(self,x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

def noise(size):
    n =  torch.randn(size,100)
    return n

def ones_target(size):
    return torch.ones(size,1)

def zeros_target(size):
    return torch.zeros(size,1)

def train_discriminator(optimizer,real_data,fake_data, discriminator, loss):
    N = real_data.size(0)
    optimizer.zero_grad()

    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real,ones_target(N))
    error_real.backward()

    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake,zeros_target(N))
    error_fake.backward()

    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer,fake_data, discriminator, loss):
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction,ones_target(N))
    error.backward()
    optimizer.step()
    return error

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def load_state(generator, discriminator, d_optim, g_optim):
    generator_state_dict = torch.load('./results/generator.pth')
    discriminator_state_dict = torch.load('./results/discriminator.pth')
    d_optim_state_dict = torch.load('./results/d_optim.pth')
    g_optim_state_dict = torch.load('./results/g_optim.pth')
    return generator.load_state_dict(generator_state_dict), discriminator.load_state_dict(discriminator_state_dict), d_optim.load_state_dict(d_optim_state_dict), g_optim.load_state_dict(g_optim_state_dict)

def save_state(generator, discriminator, d_optim, g_optim):
    torch.save(generator.state_dict(), './results/generator.pth')         #save neural network state
    torch.save(discriminator.state_dict(), './results/discriminator.pth')         #save neural network state
    torch.save(d_optim.state_dict(), './results/d_optim.pth')   #save optimizer state
    torch.save(g_optim.state_dict(), './results/g_optim.pth')   #save optimizer state


def main(argv):

    torch.manual_seed(42)           #Generate 42 manual seeds and it's "reproducible" (fixed) for every run time
    torch.backends.cudnn.enabled = False    #Turn off CUDA, so all processing are serial to make sure result reproducible

    batch_size = 100
    data_loader = mnist_data(batch_size)

    generator = Generator()
    discriminator  = Discriminator()

    d_optim = optim.Adam(discriminator.parameters(),lr = 0.0002 )
    g_optim = optim.Adam(generator.parameters(),lr = 0.0002 )

    loss = nn.BCELoss()

    num_test_samples = 16
    test_noise = noise(num_test_samples)

    load_state(generator, discriminator, d_optim, g_optim)

    num_epochs = 139

    for epoch in range(num_epochs):
        for n_batch, (real_batch, example_targets) in enumerate(data_loader):

            # train the discriminator
            real_data = real_batch.view(-1,784)

            fake_data = generator(torch.randn(batch_size,100)).detach()

            d_error,d_pred_real,d_pred_fake = train_discriminator(d_optim,real_data,fake_data, discriminator, loss)

            # traing the generator
            fake_data = generator(noise(batch_size))
            g_error = train_generator(g_optim,fake_data, discriminator, loss)

        print("Epoch: ")

    save_state(generator, discriminator, d_optim, g_optim)

    test_images = vectors_to_images(generator(test_noise))
    test_images = test_images.data

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(test_images[i][0], cmap='gray', interpolation = 'none')
        #plt.imshow(vectors_to_images(fake_data).data, cmap='gray', interpolation = 'none')
        #plt.imshow(denorm(fake_data[i].reshape((-1, 28,28)).detach())[0], cmap='gray', interpolation = 'none')
        plt.title("Latent Vector example")
        plt.xticks([])          #remove xtick
        plt.yticks([])          #remove ytick
    plt.show()

    return

if __name__ == "__main__":
    main(sys.argv)

