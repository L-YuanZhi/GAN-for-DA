import os 
import torch 
import math 
import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt 
import norm
import csv

class dataset:
    def __init__(self, img_size=100):
        self.__dataset = None
        self.__set_len = None
        self.__image_size = img_size
        self.__data_dims = None
    
    def Parameter(self):
        """
        output number and dimension of data
        and return as tuple (dims,num)
        """
        print("number of data:",self.__set_len)
        print("dimension of data:",self.__data_dims)
        print("image size:",self.__image_size)
        return (self.__data_dims,self.__set_len,self.__image_size)

    def Dataset(self, data_path, mode=None):
        """
        create an dataset for the som
        :param data_path: the directory path of the pipe images
        :param mode: normalize the image with average to 0 and variance to 1 as default with keyword "zero-one",
                     keyword "min-max" nomalize image pixel range from 0 to 1  
        :returns: an dataset contain data agumrnted images, each data will be same shape as ([flatten intensity],...)
        """
        dataset = []
        
        for dirs in os.listdir(data_path):
            for fileName in os.listdir(os.path.join(data_path,dirs)):
                input_image = plt.imread(os.path.join(data_path,dirs,fileName)) # original image for training the model
                image = cv2.resize(input_image,(self.__image_size,self.__image_size))
                self.__image_size = image.shape
                if mode == None or mode == "zero-one":
                    normalized_image = norm.Normalize_circle(image) # normalized image
                elif mode == "min-max":
                    normalized_image = norm.Normalize_circle_minmax(image)
    
                dataset.append(normalized_image)
                    
        # dataset = tuple(dataset)
        self.__set_len=len(dataset)
        self.__data_dims=len(dataset[0])

        return torch.tensor(dataset)

def dataGet(input_path,save_path):
    with open(save_path,"w") as csv_file:
        writer = csv.writer(csv_file) 
        writer.writerow(["file_name","min","max","mean"])

        d_list = os.listdir(input_path)
        for d in d_list:
            if d[-4:] != ".csv":
                image_list = os.listdir(os.path.join(input_path,d))
                for item in image_list:
                    image = plt.imread(os.path.join(input_path,d,item))

                    shape = image.shape[0]
                    c = shape/2
                    rs = 10*shape/294
                    r = c-rs 

                    aoi = []

                    for x,y in np.argwhere(image>=0):
                        if (x-c)**2+(y-c)**2<=r**2:
                            aoi.append(image[x,y])

                    m1 = np.min(aoi)
                    m2 = np.max(aoi)
                    m3 = np.mean(aoi)

                    writer.writerow([item,m1,m2,m3])

if __name__ == "__main__":
    # ip = "/home/lin/pyProject/pipe-roughness/90/mix/cut"
    ip = "images/generate_image/model_21928_bp1_none_96x96_mix"
    op = "input_tensor/model_21928_bp1_none_96x96_mix.csv"

    dataGet(ip,op)