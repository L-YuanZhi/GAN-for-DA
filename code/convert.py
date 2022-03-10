import cv2
import csv
import os 
import random

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def csv_load(path):
    # load csv file, return min\max\means\label
    csv_data = pd.read_csv(path)

    data = csv_data[["file_name","min","max","mean","norm_mean","label"]].iloc[:,:].values

    return data

def value_cvt(image_path,save_path,values,label):
    
    # cf = open("output/o%d_value_cvt_limit_log.csv" %label,"w")
    # writer = csv.writer(cf)
    # writer.writerow(["file_name","minv","maxv","mean","norm_mean","target_label"])
    for fn in os.listdir(image_path):
        img = plt.imread("%s/%s" %(image_path,fn))
        img = img/255.

        size = img.shape[0]
        rdsize = 10*size/294
        center = round(size/2)
        radius = int(center-rdsize)

        norm_mean = np.mean(img)
        r = []
        for item in values:
            if item[3]<=1.5*norm_mean and item[3] >= 0.5*norm_mean:
                r.append(item)
        
        if len(r)!=0:
            minv, maxv, meanv = r[random.randint(0,len(r)-1)][:3]
        else:
            minv, maxv, meanv = values[random.randint(0,len(values)-1)][:3]

        for x,y in np.argwhere(img>=0):
            if (x-center)**2+(y-center)**2 <= radius**2:
                if img[x,y]*(maxv-minv) + minv <= 255.:
                    img[x,y] = img[x,y]*(maxv-minv) + minv
                else:
                    img[x,y] = 255.
            else:
                img[x,y] = random.randint(5,14)
                # img[x,y] = img[x,y]*(maxv-minv)

        mean = np.mean(img)
        counter = 0

        while(mean >= 1.5*meanv or mean <= 0.5*meanv):
            if len(r)!=0:
                minv, maxv, meanv = r[random.randint(0,len(r)-1)][:3]
            else:
                minv, maxv, meanv = values[random.randint(0,len(values)-1)][:3]
            ### need norm_circle ###
            for x,y in np.argwhere(img>=0):
                if (x-center)**2+(y-center)**2 <= radius**2:
                    if img[x,y]*(maxv-minv) + minv <= 255.:
                        img[x,y] = img[x,y]*(maxv-minv) + minv
                    else:
                        img[x,y] = 255.
                else:
                    img[x,y] = random.randint(5,14)
                    # img[x,y] = img[x,y]*(maxv-minv)

            mean = np.mean(img)
            counter += 1

            if counter > 50:
                # print("eo")
                break
        
        if counter <= 50:
            cv2.imwrite("%s/%s" %(save_path,fn),img)
        #     writer.writerow(["%s/%s" %(save_path,fn),minv,maxv,mean,norm_mean,"ERROR"])   
        # else:
        #     # print("sa")
        #     writer.writerow(["%s/%s" %(save_path,fn),minv,maxv,mean,norm_mean,"k%d" %label])
            # cv2.imwrite("%s/%s" %(save_path,fn),img)
    
    # cf.close()


if __name__ == "__main__":
    cluster_iter = 0
    # image_path = [
    #     # "images/generate_image/model_211116_bp0_minmax_96x96_mix/bp0",
    #     # "images/generate_image/model_211116_bp1_minmax_96x96_mix/bp1",
    #     # "images/generate_image/model_211116_bp2_minmax_96x96_mix/bp2"
    #     "images/generate_image/model_211116_gp0_minmax_96x96_mix/gp0",
    #     "images/generate_image/model_211116_gp-1_minmax_96x96_mix/gp-1"
    # ]
    # save_path = [
    #     # "images/generate_image/cvt/OPTICS_5_circle/bp0",
    #     # "images/generate_image/cvt/OPTICS_5_circle/bp1",
    #     # "images/generate_image/cvt/OPTICS_5_circle/bp2"
    #     "images/generate_image/cvt/OPTICS_5_circle/gp0",
    #     "images/generate_image/cvt/OPTICS_5_circle/gp-1"
    # ]

    image_path = [
        "images/generate_image/model_21101_bp0_minmax_96x96_mix_CAN_increase/bp0",
        "images/generate_image/model_21101_bp1_minmax_96x96_mix_CAN_increase/bp1",
        "images/generate_image/model_21101_bp2_minmax_96x96_mix_CAN_increase/bp2"
        # "images/generate_image/model_21101_gp0_minmax_96x96_mix_CAN_increase/gp0",
        # "images/generate_image/model_21101_gp1_minmax_96x96_mix_CAN_increase/gp1"
    ]
    save_path = [
        "images/generate_image/cvt/kmeans_5_circle_2/bp0",
        "images/generate_image/cvt/kmeans_5_circle_2/bp1",
        "images/generate_image/cvt/kmeans_5_circle_2/bp2"
        # "images/generate_image/cvt/kmeans_5_circle_2/gp0",
        # "images/generate_image/cvt/kmeans_5_circle_2/gp1"
    ]
    
    csv_path = "output/KOC_bp.csv"
    data = csv_load(csv_path)
    oc = 3
    kc = 3
    c = []

    for i in range(oc):
        c.append([])
        for j in range(kc):
            c[i].append([])

    for item in data:
        # if item[5][1]=="0":
        #     o = int(item[5][1])
        # else:
        #     o = 1
        o = int(item[5][1])
        k = int(item[5][-2])
        c[o][k].append(item[:-1])
    
    # with open("output/KOC_bp_cvt.csv","w") as csv_file:
    #     writer = csv.writer(csv_file)
    #     writer.writerow(["file_name","min","max","mean","norm_mean","label"])
    #     for o in range(oc):
    #         for k in range(kc):
    #             for item in c[o][k]:
    #                 writer.writerow(item)

    col_0 = []
    col_1 = []
    col_2 = []
    for d in c:
        values = []
        for item in d:

            df = pd.DataFrame(item,columns=["file_name","min","max","mean","norm_mean"])

            if d.index(item) == 0:
                print("o0")
                for v in np.reshape(df[["min","max","mean","norm_mean"]].iloc[:,:].values,(-1,4)):
                    col_0.append(v)
            elif d.index(item) == 1:
                print("o1")
                for v in np.reshape(df[["min","max","mean","norm_mean"]].iloc[:,:].values,(-1,4)):
                    col_1.append(v)
            elif d.index(item) == 2:
                print("o2")
                for v in np.reshape(df[["min","max","mean","norm_mean"]].iloc[:,:].values,(-1,4)):
                    col_2.append(v)

            # min_values = np.reshape(df[["min"]].iloc[:,:].values,-1)
            # max_values = np.reshape(df[["max"]].iloc[:,:].values,-1)
            # mean_values = np.reshape(df[["mean"]].iloc[:,:].values,-1)
            # nm_values = np.reshape(df[["norm_mean"]].iloc[:,:].values,-1)
            
            # print("o%dk%d" %label)
            for v in np.reshape(df[["min","max","mean","norm_mean"]].iloc[:,:].values,(-1,4)):
                values.append(v)
            
            print(len(values))
        
        # ### OPTICS ###
        # value_cvt(image_path[cluster_iter],save_path[cluster_iter],values,c.index(d))
        # cluster_iter += 1

    ### K-Means ###
    value_cvt(image_path[0],save_path[0],col_0,0)
    value_cvt(image_path[1],save_path[1],col_1,1)
    if kc == 3:
        value_cvt(image_path[2],save_path[2],col_2,2)   
    # plt.hist(values)
    # plt.plot(mean_values,nm_values,"o")
    # plt.xlabel("mean")
    # plt.ylabel("norm_mean")
    # plt.show()