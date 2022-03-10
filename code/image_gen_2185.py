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

import norm

cuda = True if torch.cuda.is_available() else False

weight_info = {
    "date":2184,
    "class":"siwa",
    "normlizer":"minmax"
}

num_requier = 3000

model_name = "model_%d_%s_%s" %(weight_info["date"],weight_info["class"],weight_info["normlizer"])
# model_name = "model_2184_minmax_siwa_160"
model_date = str(weight_info["date"])
weights_num = None
config_csv = "model_configs/%s_config.csv" % model_name
# config_csv = "model_configs/model_2184_minmax_siwa_160_config.csv"
print("config csv file path:",config_csv)

with open(config_csv,"r") as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = []
    config = {}
    for row in csv_reader:
        rows.append(row)

    for i in range(len(rows[0])):
        if rows[2][i] == "'int'" :
            config[rows[0][i]] = int(rows[1][i])
        elif rows[2][i] == "'float'":
            config[rows[0][i]] = float(rows[1][i])
        elif rows[2][i] == "'str'":
            config[rows[0][i]] = rows[1][i]

n_epochs = num_requier//config["batch_size"]
last_epoch = num_requier%config["batch_size"]

if not os.path.exists("images/generate_image/%s" %model_name):
    os.mkdir("images/generate_image/%s" %model_name)

if not os.path.exists("images/generate_image/%s/%s" %(model_name,weight_info["class"])):
    os.mkdir("images/generate_image/%s/%s" %(model_name,weight_info["class"]))

#learnable parameters的载入
def weights_load(model,load_path,train_mode=False):
    model.load_state_dict(torch.load(load_path))
    if train_mode:
        model.train()
    else:
        model.eval()

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

generator = generator()

if cuda:
    generator.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

weights_list = os.listdir("model_weights/%s" %model_name)
if not  weights_num:
    nums = []
    for item in weights_list:
        item = item[item.index("_")+1:]
        item = item[item.index("_")+1:]    
        item = item[:item.index(".")]
        nums.append(int(item))
    nums.sort(reverse=True)

weights_path = "model_weights/%s/generator_%s_%d.0.pt" % (model_name,model_date,nums[0])
weights_load(generator,weights_path)
print(weights_path)

i = 0
for epoch in range(n_epochs):
    z = Variable(Tensor(np.random.normal(0,1,(config["batch_size"],config["latent_dim"]))))
    gen_imgs = generator(z)

    print(gen_imgs.size())
    for img in gen_imgs:
        plt.imsave("images/generate_image/%s/%s/%d.png" %(model_name,weight_info["class"],i),norm.Normalize_circle_minmax(img[0].cpu().detach().numpy()),cmap="Greys_r")
        i+=1

z = Variable(Tensor(np.random.normal(0,1,(last_epoch,config["latent_dim"]))))
gen_imgs = generator(z)

print(gen_imgs.size())
for img in gen_imgs:
    plt.imsave("images/generate_image/%s/%s/%d.png" %(model_name,weight_info["class"],i),norm.Normalize_circle_minmax(img[0].cpu().detach().numpy()),cmap="Greys_r")
    i+=1