import cv2
import csv
import os
import math
import random
import torch
import numpy as np 
import matplotlib.pyplot as plt

def Normalize_circle(input, t_ave=0, t_var=1):
    """
    normalize the image but only with area inside the pipe circle 
    by move the average to the similar value, default average 0 and variance 1

    :param input: the input image of the pipe
    :param t_ave: target average 
    :param t_var: target variance

    :returns: an normalized image
    """
    output = np.zeros(input.shape[:2],np.float)
    pixels = []
    average = 0

    # 获取输入图片的结构信息
    if input.shape[0]!=input.shape[1]:
        raise AttributeError("The width and height of the pipe image must be the same")
    else:
        size = input.shape[0]
        rdsize = 10*size/294
        center = round(size/2)
        radius = int(center-rdsize)

    for w,h in np.argwhere(input>=0):
        if (w-center)**2+(h-center)**2<=radius**2:
            pixels.append((w,h))
            average += input[w,h]
    average = average/len(pixels)

    variance = 0
    for w,h in pixels:
        variance += (input[w,h]-average)**2
    variance = math.sqrt(variance/len(pixels))
    
    for w,h in pixels:
        output[w,h] = t_ave + t_var*(input[w,h]-average)/variance
    
    return output

def Normalize_circle_minmax(image,start=0.,stop=1.):
    """
    normalize the image but only with area inside the pipe circle 
    by move minimum value to start and maximum to stop

    :param image: the input image of the pipe
    :param start: the minimum value of image, default as 0.
    :param stop: the maximum value of image, default as 1.

    :returns: an normalized image
    """
    if start<0 or stop>255:
        raise Warning("Result may not be save as required form")

    if image.shape[0]!=image.shape[1]:
        raise AttributeError("The width and height of the pipe image must be the same")
    else:
        size = image.shape[0]
        rdsize = 10*size/294
        center = round(size/2)
        radius = int(center-rdsize)

    output = np.zeros(image.shape[:2],np.float32)

    pmin = 255
    pmax = 0

    meanInPipe = 0.
    numInPipe = 0

    for w,h in np.argwhere(output==0):
        if (w-center)**2+(h-center**2)<=radius**2:
            if image[w,h]<pmin:
                pmin = image[w,h]
            if image[w,h]>pmax:
                pmax = image[w,h]
    
    for w,h in np.argwhere(output==0):
        if (w-center)**2+(h-center**2)<=radius**2:
            output[w,h] = ((image[w,h]-pmin)/(pmax-pmin))*(stop-start)+start
            meanInPipe += output[w,h]
            numInPipe += 1
    
    
    return output, meanInPipe/numInPipe

def execute(inPath,outPath,csvFile):
    bgDatemm = []
    gpDatemm = []
    bpDatemm = []
    bgDateac = []
    gpDateac = []
    bpDateac = []
    
    for dirs in os.listdir(inPath):
        fileList = os.listdir(os.path.join(inPath,dirs))
        for fn in fileList:
            img = plt.imread(os.path.join(inPath,dirs,fn))

            resize_image = cv2.resize(img,(96,96),interpolation=cv2.INTER_AREA)
            
            ac_img = Normalize_circle(resize_image)
            mm_img = Normalize_circle_minmax(resize_image)
            
            bgDateac.append(ac_img)
            bgDatemm.append(mm_img)

            if dirs == "bp":
                bpDateac.append(ac_img)
                bpDatemm.append(mm_img)
                # csvFile.write(fn+","+"0\n")
            elif dirs == "gp":
                gpDateac.append(ac_img)
                gpDatemm.append(mm_img)
                # csvFile.write(fn+","+"1\n")

    # for fn in os.listdir(inPath):
    #     img = plt.imread(os.path.join(inPath,fn))
    #     img_resize = cv2.resize(img,(192,192))
    #     # ac_img = Normalize_circle(img_resize)
    #     mm_img = Normalize_circle_minmax(img_resize)
    #     # bgDateac.append(ac_img)
    #     # bgDatemm.append(mm_img)


    #     bpDatemm.append(mm_img)


    torch.save(torch.tensor(bgDateac),outPath+"/bg_stdn_96x96_rs_area_mix.pt")
    torch.save(torch.tensor(bpDateac),outPath+"/bp_stdn_96x96_rs_area_mix.pt")
    torch.save(torch.tensor(gpDateac),outPath+"/gp_stdn_96x96_rs_area_mix.pt")
    torch.save(torch.tensor(bgDatemm),outPath+"/bg_minmax_96x96_rs_area_mix.pt")
    torch.save(torch.tensor(bpDatemm),outPath+"/bp_minmax_96x96_rs_area_mix.pt")
    torch.save(torch.tensor(gpDatemm),outPath+"/gp_minmax_96x96_rs_area_mix.pt")

def _execute(inPath,outPath,config):
    
    dList = os.listdir(inPath)
    
    for d in dList:
        dataset = []

        for item in os.listdir("%s/%s" %(inPath,d)):
            image = plt.imread("%s/%s/%s" %(inPath,d,item))

            resizeImage = cv2.resize(image,(config["size"],config["size"]),interpolation=cv2.INTER_AREA)

            if config["norm"]=="minmax":
                normImage = Normalize_circle_minmax(resizeImage)
            elif config["norm"]== "stdn":
                normImage = Normalize_circle(resizeImage)
            elif config["norm"]=="none":
                normImage = resizeImage

            # resizeImage = cv2.resize(image,(config["size"],config["size"]),interpolation=cv2.INTER_AREA)
            
            dataset.append(normImage)
        
        className = "%s%s_%s_%dx%d_%s.pt" %(config["label"],d,config["norm"],config["size"],config["size"],config["source"])

        torch.save(torch.tensor(dataset),"%s/%s" %(outPath,className))

def exc_result_img(inPath,outPath,config):
    dList = os.listdir(inPath)
    
    csvFile = open("%s/value.csv" %outPath,"w")
    writer = csv.writer(csvFile)
    writer.writerow(["file_name","min","max","mean","mean_in_pipe"])
    cfbp = open("%s/bp_value.csv" %outPath,"w")
    writer_bp = csv.writer(cfbp)
    writer_bp.writerow(["file_name","min","max","mean","mean_in_pipe"])
    cfgp = open("%s/gp_value.csv" %outPath,"w")
    writer_gp = csv.writer(cfgp)
    writer_gp.writerow(["file_name","min","max","mean","mean_in_pipe"])

    for d in dList:
        
        if not os.path.exists("%s/%s" %(outPath,d)):
            os.mkdir("%s/%s" %(outPath,d))

        for item in os.listdir("%s/%s" %(inPath,d)):
            image = plt.imread("%s/%s/%s" %(inPath,d,item))

            resizeImage = cv2.resize(image,(config["size"],config["size"]),interpolation=cv2.INTER_AREA)
            
            if config["norm"]=="minmax":
                normImage,meanInPipe = Normalize_circle_minmax(resizeImage)
            elif config["norm"]== "stdn":
                normImage = Normalize_circle(resizeImage)
            elif config["norm"]=="none":
                normImage = resizeImage
            
            writer.writerow([item,np.min(normImage),np.max(normImage),np.mean(normImage),meanInPipe])

            if item[:2] == "NG":
                writer_bp.writerow([item,np.min(normImage),np.max(normImage),np.mean(normImage),meanInPipe]) 
            else:
                writer_gp.writerow([item,np.min(normImage),np.max(normImage),np.mean(normImage),meanInPipe])
            
            cv2.imwrite("%s/%s/%s" %(outPath,d,item),255.*normImage)
    
    csvFile.close()

def PipeRotation(inputImage,angle,radius= 0):
    size = inputImage.shape[0]
    center = round(size/2)
    if not radius:
        rdsize = 10*size/294
        radius = int(center-rdsize)

    mat = cv2.getRotationMatrix2D((center,center),angle,1)
    return cv2.warpAffine(inputImage,mat,(96,96))

def IncreaseDataset(inPath,outPath,config):

    target_sample_num = 320

    for d in os.listdir(inPath):
        smaple_num = len(os.listdir("%s/%s" %(inPath,d)))
        increase_num = round(target_sample_num/smaple_num)
        ds = []
        for f in os.listdir("%s/%s" %(inPath,d)):
            image = plt.imread("%s/%s/%s" %(inPath,d,f))
            image = cv2.resize(image/255.,(96,96),interpolation=cv2.INTER_AREA)
            for i in range(increase_num):
                angle = random.randint(0,360)
                ds.append(PipeRotation(image,angle))
            
        torch.save(torch.tensor(ds),"%s/%s%s_%s_%dx%d_%s_%s.pt" %(outPath,config["label"],d,
            config["norm"],config["size"],config["size"],config["source"],config["other"]))

def ToTensorSet(inPath,outPath,config):
    
    for d in os.listdir(inPath):
        ds = []

        for item in os.listdir("%s/%s" %(inPath,d)):
            img = plt.imread("%s/%s/%s" %(inPath,d,item))
            img = img/255.
            ds.append(img)
        
        torch.save(torch.tensor(ds),"%s/%s%s_%s_%dx%d_%s_%s.pt" %(outPath,config["label"],d,
            config["norm"],config["size"],config["size"],config["source"],config["other"]))

if __name__=="__main__":
    # inPath0= "/home/lin/pyProject/pipe-roughness/90/mix/cut"
    # outPath0 = "output/minmax_96x96_mix"
    inPath = "output/cluster_results/minmax_96x96_mix/"
    # inPath = "output/minmax_96x96_mix/mix_bp"
    outPath = "input_tensor"
    # csvFile = open("input_tensor/bg_label_none_203.csv","w")
    # execute(inPath0,outPath,None)
    # csvFile.close()

    config = {
        "norm":"minmax",
        "size":96,
        "source":"mix",
        "label":"un",
        "other":"UN_increase"
    }

    # _execute(inPath,outPath,config)
    # exc_result_img(inPath0,outPath0,config)
    # ToTensorSet(inPath,outPath,config)
    IncreaseDataset(inPath,outPath,config)
