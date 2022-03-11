from __future__ import print_function

import os
import csv
import cv2
import random
import torch

import torch.nn as nn 
import torch.nn.parallel
# import torch.nn.DataParallel #由于GPU平行学习
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as dset 
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

from IPython.display import HTML 
from torchvision.utils import save_image

# https://github.com/eriklindernoren/PyTorch-GAN
# を参考　
# このプログラムはまだ調整が必要！！！


config = {
    "model_name":None,#"model_21727_ReLU",
    "latent_dim":100,
    "channels":1,
    "img_size":96,
    "n_epochs":1000,
    "batch_size":16,
    "b1":0.5,
    "b2":0.999,
    "lr":0.005,
    "sample_interval":400,
    "real_img":"bp_minmax_96x96_rs_area_mix.pt"
}

#Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1,1000) #use if you want new results
print("Random Seed: ",manualSeed)
random.seed(manualSeed)
#torch.manual_seed(int) 在需要生成随机数的实验中，只要输入的int不变，则生成的随机数都是固定的
torch.manual_seed(manualSeed)

#inputs 全局变量设置
#Root directory for dataset
dataroot = "/home/lin/pyProject/pipe-roughness/90/bp157/cut"
#Number of workers for dataloader
workers = 2
#Batch size during training
batch_size = config["batch_size"]
#Spatial size of training images. All images will be resized to this size using a transformer
image_size = config["img_size"]
#Number of channels in the training images. For color images this is 3
nc = 1
#Size of z latent vector (i.e. size of generator input)
nz = 100
#Size of feature maps in generator
ngf = 64
#Size of feature maps in discriminator
ndf = 64
#Number of training epochs
num_epochs = 5
#Learing rate for optimizers
lr = 0.0002
#Bata1 hyperparam for Adam optimizers
beta1 = 0.5
#Number of GPUs avialabel. Use 0 for CPU mode.
ngpu = 1

#We can use an image folder dataset the way we have it setup
#Create the dataset
# dataset = dset.ImageFolder(
#     root=dataroot,
#     transform=transforms.Compose([
#         transforms.Resize(image_size),
#         transforms.CenterCrop(image_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
#     ])
# )
#Create the dataloader
# dataloader = data.DataLoader(
#     dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=workers
# )
dataloader = data.DataLoader(
    torch.load(os.path.join("input_tensor",config["real_img"])),
    batch_size=config["batch_size"],
    shuffle=True
)

#Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#Plot some training images
real_batch = next(iter(dataloader))
# plt.figure(figsize=(4,4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device),
#     padding=2,normalize=True).cpu(),(1,2,0)))
# plt.show()

#Custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        nn.init.normal_(m.weight.data,0.0,0.2)
    elif classname.find("BatchNorm")!=-1:
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

#Generator
class Generator(nn.Module):
    def __init__(self,ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is Z, going into a convolution
            nn.ConvTranspose2d(nz,ngf*16,3,1,0,bias=False),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(True),
            #state size. (ngf*8)*4*4

            nn.ConvTranspose2d(ngf*16,ngf*8,3,2,1,bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8,ngf*4,3,2,1,bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            #state size. (ngf*4)*8*8

            nn.ConvTranspose2d(ngf*4,ngf*2,3,2,1,bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            #state size. (ngf*2)*16*16

            nn.ConvTranspose2d(ngf*2,ngf,3,2,1,bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf,nc,3,2,1,bias=False),
            nn.ReLU(True)
            # nn.Tanh()
            #state size. (nc)*64*64
        )

    def forward(self,input):
        return self.main(input)
        #ERROR running_mean should contain 64 elements not 128

#Create the generator
netG = Generator(ngpu).to(device)

#Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netG.apply(weights_init)

#Print the model
print(netG)

#Discriminator
class Discriminator(nn.Module):
    def __init__(self,ngpu):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #input is (nc)*64*64
            nn.Conv2d(nc,ndf,3,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            #state size. (ndf)*32*32
            nn.Conv2d(ndf,ndf*2,3,2,1,bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2,inplace=True),

            #state size. (ndf*2)*16*16
            nn.Conv2d(ndf*2,ndf*4,3,2,1,bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(.2,inplace=True),

            #state size. (ndf*4)*8*8
            nn.Conv2d(ndf*4,ndf*8,3,2,1,bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2,inplace=True),

            nn.Conv2d(ndf*8,ndf*16,3,2,1,bias=False),
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2,inplace=True),

            #state size. (ndf*8)*3*3
            nn.Conv2d(ndf*16,1,3,1,0,bias=False),
            nn.Sigmoid()
        )

    def forward(self,input):
        return self.main(input)
        #Calculated padded input size per channel: (2 x 2). Kernel size: (4 x 4).
        #Kernel size can't be greater than actual input size
        #Given groups=1, weight of size [64, 3, 4, 4], 
        #expected input[128, 64, 32, 32] to have 3 channels, but got 64 channels instead

#Create the Discriminator
netD = Discriminator(ngpu).to(device)

#Handle multi-gpu if desired
if (device.type == "cuda") and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
netD.apply(weights_init)

#Print the model
print(netD)

#Initialize BECLoss function
criterion = nn.BCELoss()

#Create batch of latent vectors taht we will use to visualize the progression of the generator
fixed_noise = torch.randn(16,nz,1,1,device=device)

#Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

#Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(),lr=lr,betas=(beta1,0.999))
optimizerG = optim.Adam(netG.parameters(),lr=lr,betas=(beta1,0.999))

#Training Loop

#Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
#For each epoch
for epoch in range(num_epochs):
    #For each batch in dataloader
    for i, img in enumerate(dataloader,0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        img = img.view(-1,nc,96,96)
        # print("img",img.shape)
        #Format batch
        real_cpu = img.to(device)
        # real_cpu 是输入的图片 [96,96]
        # print(real_cpu.size())
        b_size = real_cpu.size(0)
        label = torch.full((b_size,),real_label,dtype=torch.float,device=device)
        #Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        #Calculate loss on all-real batch
        # print(output.size())
        # print(output)
        errD_real = criterion(output,label)
        # Exception has occurred: ValueError
        # target size (torch.Size([128]))  different to input size (torch.Size([3200])).

        #Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ##Train with all-fake batch 
        #Generate batch of latent vectors
        noise = torch.randn(b_size,nz,1,1,device=device)
        #Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        #Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        #Calculate D's loss on the all-fake batch
        errD_fake = criterion(output,label)
        #Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        #Compute error of D as sum over the fake and the real batches
        errD = (errD_real + errD_fake)/2.
        #Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label) # fake labels are for generator cost
        #Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        #Calculate G's loss based on this output
        errG = criterion(output,label)
        #Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        #Update G
        optimizerG.step()

        #Output training stats
        if i % 50 == 0:
            print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f/%.4f"
            % (epoch,num_epochs,i,len(dataloader),errD.item(),errG.item(),D_x,D_G_z1,D_G_z2))

        #Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        #Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake,padding=2,normalize=True))

            save_path = "%s/%d.bmp" % ("for_test/generate_image",epoch)
            save_image(255.*fake.data[:16],save_path,nrow = 4, normalize=True)

            # for img in fake.numpy():
            #     # plt.imsave("images/generate_image/%s/%s/%d.png" %(model_name,weight_info["class"],i),norm.Normalize_circle_minmax(img[0].cpu().detach().numpy()),cmap="Greys_r")
            #     gen_image, _ = norm.Normalize_circle_minmax(img[0].cpu().detach().numpy())
            #     cv2.imwrite("images/generate_image/%s/%s/%d.bmp" %(model_name,weight_info["class"][i_num],i),255.*gen_image)
            #     writer.writerow(torch.reshape(img,(-1,)).cpu().detach().numpy())
            #     i+=1
        iters += 1

with open("for_test/model_loss.csv","w") as c_file:
    writer =  csv.writer(c_file)
    writer.writerow(["epoch","G_loss","D_loss"])
    for i in range(num_epochs):
        writer.writerow([i+1,G_losses[i],D_losses[i]])

# fig = plt.figure(figsize=(4,4))
# plt.axis("off")
# ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
# ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
# plt.show()

# HTML(ani.to_jshtml())