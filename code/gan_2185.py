import os
import numpy as np 
import math
import time
import csv 

import torch
import torch.nn as nn
import torch.nn.functional as functional 
import matplotlib.pyplot as plt 

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torchvision.transforms as transforms
from torchvision.utils import save_image

cuda = True if torch.cuda.is_available() else False

config = {
    "model_name":None,#"model_21727_ReLU",
    "latent_dim":100,
    "channels":1,
    "img_size":160,
    "n_epochs":1000,
    "batch_size":16,
    "b1":0.5,
    "b2":0.999,
    "lr":0.005,
    "sample_interval":400,
    "real_img":"minmax_siwa_160.pt"
}

#获取时间日期
def date():
    lt= time.localtime(time.time())
    return "_"+str(lt[0])[-2:]+str(lt[1])+str(lt[2])+"_"

#权重初始化
def weights_init_normal(m):
    #获取模组名称, classname是字符串
    classname = m.__class__.__name__
    
    #对卷积层和BN层进行初始化
    # str.find() 会在str中查找目标字符串，不存在则返回-1
    if classname.find("Conv") != -1:
        # nn.init.normal_(tensor,mean=0.,std=1.) 从给定的正态分布N(mean,std^2)中生成值填充至tensor
        torch.nn.init.normal_(m.weight.data,0.0,0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data,1.0,0.02)
        # nn.init.constant_(tensor,val) 以val填充tensor
        torch.nn.init.constant_(m.bias.data,0.0)

#learnable parameters(weights/bias)的存储
def weights_save(model,optimizer,save_path,date,batches_done):
    classname = model.__class__.__name__
    print("Class name:",classname)
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor,"\t",model.state_dict()[param_tensor].size())

    path = os.path.join(save_path,classname+date+str(batches_done)+".pt")
    torch.save(model.state_dict(),path)    

    print("Optimizer's state_dict:")
    for var_name in optimizer  .state_dict():
        print(var_name,"\t",optimizer.state_dict()[var_name])

#learnable parameters的载入
def weights_load(model,load_path,train_mode=False):
    model.load_state_dict(torch.load(load_path))
    if train_mode == True:
        model.train()
    else:
        model.eval()

#生成器类
class generator(nn.Module):
    #类初始化
    def __init__(self):
        # super(class,self)函数会找到class的父类，在将子类对象转换父类对象
        super().__init__()

        self.init_size = config["img_size"] // 4 # 输入图片的尺寸
        # 建立线性模型。nn.Linear(batch_size,size)对输入进行线性转换。y=xAt+b。batch_size可视为输入的样本数。
        # 常用作全链接层，size是输入权重的次元数，out_size是输出权重的次元数，即分类的class数。
        self.l1 = nn.Sequential(nn.Linear(config["latent_dim"],128*self.init_size**2))

        #卷积功能模块
        self.conv_blocks = nn.Sequential(
            # nn.BatchNorm2d(num_features,eps)对权重进行BN正则化，num_features与权重的channel数是等价的。
            # eps是函数的分母的参数，默认为1e-5。
            nn.BatchNorm2d(128),
            # nn.Upsample(scale_factor)，scale_factor输出为输入的多少倍，可以理解为上采样放大边界尺寸的倍数。
            nn.Upsample(scale_factor=2),
            # nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,...)
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.BatchNorm2d(128,0.8),
            # nn.LeakyReLU(negative_slope,inplace) negative_slope是小于零的部分的直线曲率，inplace=true会覆盖原先的张量，只要不导致错误打开可以节省内存。
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,stride=1,padding=1),
            nn.BatchNorm2d(64,0.8),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.ReLU(True),
            nn.Conv2d(64,config["channels"],3,stride=1,padding=1),
            nn.BatchNorm2d(config["channels"]),
            # nn.LeakyReLU(0.1)
            nn.ReLU(True)
            # nn.Tanh()
        )

    def forward(self,z):
        out = self.l1(z)
        out = out.view(out.shape[0],128,self.init_size,self.init_size)
        # print("生成器l1层输出大小:",out.shape)
        img = self.conv_blocks(out)
        # print("生成器输出大小:",img.shape)
        return img


#识别器类
class discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                # nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
                nn.Conv2d(in_filters,out_filters,3,2,1),
                nn.LeakyReLU(0.2,inplace=True),
                nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters,0.8))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(config["channels"],16,bn=False),
            *discriminator_block(16,32),
            *discriminator_block(32,64),
            *discriminator_block(64,128)         
        )

        ds_size = config["img_size"] //2**4 # 卷积后的特征图单边长
        self.adv_layer = nn.Sequential(nn.Linear(128*ds_size**2,1), nn.Sigmoid())

    def forward(self,img):
        # print("识别器输入大小:",img.shape)
        out = self.model(img.view(img.shape[0],1,config["img_size"],config["img_size"])) # shape '[16, 1, 256, 256]' is invalid for input of size 147456
        # print("识别器输出大小:",out.size())
        out = out.view(out.shape[0],-1) # flatten()
        # print("识别器输出次元:",out.shape[1])
        # out = out.view(-1,128*ds_size**2)
        validity = self.adv_layer(out)

        return validity

#损失函数
adversarial_loss = nn.BCELoss()

#初始化G/D模型
generator = generator()
discriminator = discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

#权重初始化
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

#数据输入
dataloader = DataLoader(
    torch.load(os.path.join("input_tensor",config["real_img"])),
    batch_size=config["batch_size"],
    shuffle=True
)

#优化器
learn_rate = config["lr"]
b1 = config["b1"]
b2 = config["b2"]
optimizer_G = torch.optim.Adam(generator.parameters(),lr=learn_rate,betas=(b1,b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(),lr=learn_rate,betas=(b1,b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#weights save/load path and settings
dates = date()
if config["model_name"] == None:
    config["model_name"] = "model%s%s" %(dates,config["real_img"][:-3])
csv_file_path = "model_configs/%s_config.csv" %config["model_name"]
model_structure_path =  "model_configs/%s_structure.csv" %config["model_name"]
images_path = "images/%s" %config["model_name"]
weights_path = "model_weights/%s" %config["model_name"]

if not os.path.exists(images_path):
    os.mkdir(images_path)
if not os.path.exists(weights_path):
    os.mkdir(weights_path)


with open(csv_file_path,"w") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(config.keys())
    param_dtypes = []
    for value in config.values():
        param_dtypes.append(str(type(value))[7:-1])
    csv_writer.writerow(config.values())
    csv_writer.writerow(param_dtypes)

with open(model_structure_path,"w") as msp:
    csv_writer = csv.writer(msp)
    csv_writer.writerow("Generator")
    for param_tensor in generator.state_dict():
        csv_writer.writerow([param_tensor,generator.state_dict()[param_tensor].size()])
    csv_writer.writerow("Discriminator")
    for param_tensor in discriminator.state_dict():
        csv_writer.writerow([param_tensor,discriminator.state_dict()[param_tensor].size()])

# training
for epoch in range(config["n_epochs"]):
    for i,imgs in enumerate(dataloader):
        
        # print("imgs.i",i)
        # 
        valid = Variable(Tensor(imgs.shape[0],1).fill_(1.0),requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0],1).fill_(0.0),requires_grad=False)

        #
        real_imgs = Variable(imgs.type(Tensor))

        #Train Generator

        optimizer_G.zero_grad()

        #
        z = Variable(Tensor(np.random.normal(0,1,(imgs.shape[0],config["latent_dim"]))))
        # print("生成器输入大小:",z.shape)
        gen_imgs = generator(z)
        # print("生成器输出图片大小:",gen_imgs.shape)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        #Train Discriminator

        optimizer_D.zero_grad()

        #
        real_loss = adversarial_loss(discriminator(real_imgs),valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()),fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            %(epoch, config["n_epochs"], i, len(dataloader), d_loss.item(),g_loss.item())
        )

        batches_done = (epoch * len(dataloader) + i) 
        if batches_done % config["sample_interval"] == 0:
            save_path = "%s/%d.bmp" % (images_path,batches_done / config["sample_interval"])
            save_image(gen_imgs.data[:16],save_path,nrow = 4, normalize=True)
            weights_save(generator,optimizer_G,weights_path,dates,batches_done / config["sample_interval"])
        