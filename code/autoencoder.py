import os
import cv2
import torch

import torch.nn as nn
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

    def __encoder(self):
        
        self.__E = nn.Sequential(
        nn.Conv2d(1,10,5),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),

        nn.Conv2d(10,30,5),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True),

        nn.Conv2d(30,10,5),
        nn.MaxPool2d(2),
        nn.ReLU(inplace=True)
    )

            

    def __decoder(self):

    def feature_ouput(self):

    def orward(self):
