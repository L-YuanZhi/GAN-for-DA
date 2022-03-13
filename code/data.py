import os 
import torch 
import math 
import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt 
import norm
import csv

class dataset:
    #フォルダから画像を入力して，処理した後にtensorとして保存．
    def __init__(self, img_size=100):
        """
        initital function
        :param img_size: all image will resized to this value.
        """
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
        create an dataset
        :param data_path: the directory path of the pipe images
        :param mode: normalize the image with average to 0 and variance to 1 as default with keyword "zero-one",
                     keyword "min-max" nomalize image pixel range from 0 to 1  
        :returns: an dataset contain data agumrnted images, each data will be same shape as ([flatten intensity],...)
        """
        dataset = []
        
        for dirs in os.listdir(data_path):
            for fileName in os.listdir(os.path.join(data_path,dirs)):
                input_image = plt.imread(os.path.join(data_path,dirs,fileName))
                # original image for training the model
                
                image = cv2.resize(input_image,(self.__image_size,self.__image_size))
                # resize the image to img_size
                
                self.__image_size = image.shape
                
                if mode == None or mode == "zero-one":
                    # normalize images
                    normalized_image = norm.Normalize_circle(image) # normalized image
                elif mode == "min-max":
                    # normalize images
                    normalized_image = norm.Normalize_circle_minmax(image)
    
                dataset.append(normalized_image)
                    
        # dataset = tuple(dataset)
        self.__set_len=len(dataset)
        self.__data_dims=len(dataset[0])

        return torch.tensor(dataset)

def dataGet(input_path,save_path):
    """
    get the min/max/mean values of images, and save as CSV file.
    :param input_path: the path of directory witch image are stored.
    :param save_path: csv file name or path.
    """
    with open(save_path,"w") as csv_file:
        writer = csv.writer(csv_file) 
        writer.writerow(["file_name","min","max","mean"])

        d_list = os.listdir(input_path)
        for d in d_list:
            if d[-4:] != ".csv":
                image_list = os.listdir(os.path.join(input_path,d))
                for item in image_list:
                    # loading image from directory
                    image = plt.imread(os.path.join(input_path,d,item))
                    
                    #calculate the pipe area.
                    shape = image.shape[0]
                    c = shape/2
                    rs = 10*shape/294
                    r = c-rs 

                    aoi = []# aoi means area of interesting
                    
                    #collecting the density value of pixels inside the pipe arae
                    for x,y in np.argwhere(image>=0):
                        if (x-c)**2+(y-c)**2<=r**2:
                            aoi.append(image[x,y])

                    m1 = np.min(aoi)
                    m2 = np.max(aoi)
                    m3 = np.mean(aoi)

                    # write the values into the csv file
                    writer.writerow([item,m1,m2,m3])

if __name__ == "__main__":
    # ip = "/home/lin/pyProject/pipe-roughness/90/mix/cut"
    ip = "images/generate_image/model_21928_bp1_none_96x96_mix"
    op = "input_tensor/model_21928_bp1_none_96x96_mix.csv"

    dataGet(ip,op)
