import os
import numpy as np 
import math
import time
import csv 
import cv2

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
# PS:中国語の部分はpytorchの関数についての説明．
#    必要であれば公式ドキュメントを確認してください．

weight_info = {
    # 使用するGANモデルの重み情報．
    # 重みの読み込みや，生成画像の保存などに関連する．
    "date":21119,
    "class":["bp0","bp1","bp2","gp0","gp1"],
    "normlizer":"minmax"
}

num_requier = 3000 # 生成する画像の数．
model_nums = 5 # 使用するモデルの数．単一モデルをしようする場合は1．0にすると画像生成はしない．
for i_num in range(model_nums):
    # 5クラスのがぞうをまとめて生成する．
    
    # i_num = 4 # model_name_list中の特定なモデルをしようしたいときに，model_numsを1にしてからこの行を有効にする．

    model_name_list = [
        # 使用したモデルのリスト
        "model_211116_bp0_minmax_96x96_mix",
        "model_211116_bp1_minmax_96x96_mix",
        "model_211116_bp2_minmax_96x96_mix",
        "model_211116_gp0_minmax_96x96_mix",
        "model_211116_gp-1_minmax_96x96_mix"
    ]

    # model_name = "model_%d_%s_%s" %(weight_info["date"],weight_info["class"],weight_info["normlizer"])
    # weight_infoによって自動的にモデル名を生成する．格式が違うと使用できない．

    # model_name = "model_21927_gp-1_minmax_96x96_mix"
    # 特定なモデルをしようしたときに．

    model_name = model_name_list[i_num]
    # model_date = str(weight_info["date"])
    model_date = model_name[6:12]

    # weights_num = [16,25,14,26,22][i_num]
    weights_num = None
    # 重み読み込み用，Noneである場合モデルフォルダないを番号が一番大きいおもみを読み込み．

    config_csv = "model_configs/%s_config.csv" % model_name
    # config_csv = "model_configs/model_2184_minmax_siwa_160_config.csv"
    # 学習時のGANモデルのパラメータ．

    print("config csv file path:",config_csv)
    
    with open(config_csv,"r") as csv_file:
        # 学習時のGANモデルのパラメータを読み込み．
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
        # 生成する画像を保存するためのフォルダが存在するかを確認
        os.mkdir("images/generate_image/%s" %model_name)

    if not os.path.exists("images/generate_image/%s/%s" %(model_name,weight_info["class"][i_num])):
        # 生成する画像を保存するためのフォルダが存在しない場合，自動的に生成する．
        os.mkdir("images/generate_image/%s/%s" %(model_name,weight_info["class"][i_num]))

    # モデルの重みを読み込み
    def weights_load(model,load_path,train_mode=False):
        model.load_state_dict(torch.load(load_path))
        if train_mode:
            model.train()
        else:
            model.eval()

    class generator(nn.Module):
        # 生成器構築．
        # 使用したモデルの生成器と全く同じである必要がある．
        # GANモデルの構造が変更したら変更する必要がある．

        def __init__(self):
            # super(class,self)函数会找到class的父类，在将子类对象转换父类对象
            super().__init__()

            self.init_size = config["img_size"] // 4 # 入力がぞうのサイズ
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

    if not  weights_num:
        # weight_numが指定しない場合モデルフォルダないを番号が一番大きいおもみを読み込み．
        weights_list = os.listdir("model_weights/%s" %model_name)
        nums = []
        for item in weights_list:
            item = item[item.index("_")+1:]
            item = item[item.index("_")+1:]    
            item = item[:item.index(".")]
            nums.append(int(item))
        nums.sort(reverse=True)
        weights_num = nums[0]

    weights_path = "model_weights/%s/generator_%s_%d.0.pt" % (model_name,model_date,weights_num)
    weights_load(generator,weights_path)
    print(weights_path)

    i = 0
    for epoch in range(n_epochs):
        # 画像生成
        z = Variable(Tensor(np.random.normal(0,1,(config["batch_size"],config["latent_dim"]))))
        gen_imgs = generator(z)

        print(gen_imgs.size())
        for img in gen_imgs:
            # plt.imsave("images/generate_image/%s/%s/%d.png" %(model_name,weight_info["class"],i),norm.Normalize_circle_minmax(img[0].cpu().detach().numpy()),cmap="Greys_r")
            gen_image, _ = norm.Normalize_circle_minmax(img[0].cpu().detach().numpy())
            # 正規化
            cv2.imwrite("images/generate_image/%s/%s/%d.bmp" %(model_name,weight_info["class"][i_num],i)
                        ,255.*gen_image)
            # .bmpは1より小さい値を0にする．正常に画像を保存するために画像を255倍にする．
            i+=1

# z = Variable(Tensor(np.random.normal(0,1,(last_epoch,config["latent_dim"]))))
# gen_imgs = generator(z)

# print(gen_imgs.size())
# for img in gen_imgs:
#     # plt.imsave("images/generate_image/%s/%s/%d.png" %(model_name,weight_info["class"],i),norm.Normalize_circle_minmax(img[0].cpu().detach().numpy()),cmap="Greys_r")
#     cv2.imwriter("images/generate_image/%s/%s/%d.bmp" %(model_name,weight_info["class"],i),norm.Normalize_circle_minmax(img[0].cpu().detach().numpy()))
#     i+=1