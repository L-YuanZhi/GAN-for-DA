import cv2
import os
import csv 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 

def CircleNorm(img,value,wor):

    size = img.shape[0]
    center = round(size/2)
    reduceValue = 10 * size/294
    radius = center - reduceValue

    mean = 0
    counter = 0

    for x,y in np.argwhere(img >= 0):
        # if (x-center)**2 + (y-center)**2 < radius**2 and (x-center)**2 + (y-center)**2 > (radius-wor)**2:
        if(x-center)**2 + (y-center)**2 > (center+ wor)**2:
            mean = mean + img[x,y]
            counter = counter + 1
    
    mean /= counter

    for x,y in np.argwhere(img>=0):
        if img[x,y] -mean +value < 0:
            img[x,y] = 0
        elif img[x,y] -mean +value > 255:
            img[x,y] = 255
        else:
            img[x,y] = img[x,y] -mean +value

    return img



if __name__ == "__main__":
    # loadPath = "output/cluster_results/mix_gp_96x96"
    loadPath = "images/generate_image/cvt/OPTICS_5_circle"
    savePath = "output/cluster_results/circleNorm"

    if not os.path.exists("%s/%s" %(savePath,loadPath[-15:])):
            os.mkdir("%s/%s" %(savePath,loadPath[-15:]))

    for d in os.listdir(loadPath):
        if d.endswith(".txt"):
            continue
        if not os.path.exists("%s/%s/%s" %(savePath,loadPath[-15:],d)):
            os.mkdir("%s/%s/%s" %(savePath,loadPath[-15:],d))

        for fn in os.listdir("%s/%s" %(loadPath,d)):
            img = plt.imread("%s/%s/%s" %(loadPath,d,fn))

            img = CircleNorm(img,10,10)
            cv2.imwrite("%s/%s/%s/%s" %(savePath,loadPath[-15:],d,fn),img)