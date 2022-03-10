import os 
import cv2
import csv

import numpy as np 
import matplotlib.pyplot as plt 


## Path of input images and output images

# loadPath = "output/cluster_results/mix_bp"
loadPath = "output/cluster_results/mix_gp"
# savePath = "output/cluster_results/fft/divideFineBase"
savePath = "output/cluster_results/fft/seprate_mask_plot"


def circle(size,r2,r1=0):
    """
    Return two masks. Shape a ring, which radius between r1 and r2.
    ring zone of f_mat_0 value 1.0, others value 0.0.
    ring zone of f_mat_1 value 0.0, others value 1.0.
    """
    f_mat_1 = np.ones((size,size),dtype=float)
    f_mat_0 = np.zeros((size,size),dtype=float)

    c = size/2.

    for x,y in np.argwhere(f_mat_1==1):
        if (c-x)**2+(c-y)**2 < r2**2 and (c-x)**2+(c-y)**2 >= r1**2:
        # if (48-x)**2+(48-y)**2 > 2**2:
            # if x>40 and x<56:
                f_mat_0[x,y]=1.
                f_mat_1[x,y]=0.
                # fshift[x,y] = 0
    
    return f_mat_0,f_mat_1

def value(size,fshift,b=0):
    """
    Return two masks. 
    f_mat_0 contains points which value of fshift is bigger than b.
    f_mat_1 contains points which value of fshift is smaller than b.
    """
    f_mat_1 = np.ones((size,size),dtype=float)
    f_mat_0 = np.zeros((size,size),dtype=float)

    c = size/2.
    for x,y in np.argwhere(fshift>=b):
        # if fshift[x,y] <= a:
            f_mat_0[x,y] = 1.
            f_mat_1[x,y] = 0.
    
    return f_mat_0,f_mat_1

# d_mat = np.ones((96,96),dtype=float)
# for d in range(3,48,6):
#     d0,d1 = circle(96,d,d-3)
#     d_mat = d_mat*d1

def bAndN(ms,image):
    """
    Return normalized magnitude_spectrum (mst) and it's threshold results
    """
    
    fo = np.fft.fft2(image)
    fshifto = np.fft.fftshift(fo)
    mso = 20*np.log(np.abs(fshifto))
    if np.max(mso) != np.inf:
        maxv = np.max(mso)
    else:
        maxv = 255

    if np.min(mso) != -np.inf:
        minv = np.min(mso)
    else:
        minv=0

    mst = np.ones((96,96),dtype=np.uint8)
    for x,y in np.argwhere(mst ==1):
        mst[x,y] = (255.*(ms[x,y]-minv)/(maxv-minv))

    th = cv2.adaptiveThreshold(mst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)
    return th,mst

def seprate(dmask,pipeMask):
    """
    Return defect mask, which is the result of pipe image divide by fineBase.
    """
    mean_inside = 0
    counter = 0
    for x,y in np.argwhere(dmask>=np.min(dmask)):
        if (x-48)**2+(y-48)**2<44**2 and (x-48)**2+(y-48)**2>40**2:
            mean_inside+=dmask[x,y]
            counter+=1
    mean_inside /= counter
    # print(mean_inside)

    mean_outside = 0
    counter = 0
    for x,y in np.argwhere(pipeMask==0):
        # if (x-48)**2+(y-48)**2 >48**2:
        mean_outside += dmask[x,y]
        counter += 1
    mean_outside /= counter
    # print(mean_outside)

    for x,y in np.argwhere(pipeMask>0):
        # if (x-48)**2+(y-48)**2<46**2:
        dmask[x,y] += (mean_outside - mean_inside )
        # else:
    # for x,y in np.argwhere(pipeMask==0):
    #     dmask[x,y] = mean_outside
    
    return dmask

def dist(image,pipeMask):
    """
    Return set of pixel values, and other set of distribution of pixel values.
    """
    value = []
    number = []
    for x,y in np.argwhere(pipeMask>0):
        v = round(1000*image[x,y])
        if v not in value:
            value.append(v)
            number.append(1)
        else:
            number[value.index(v)] += 1
    value = np.array(value)
    number = np.array(number)
    # print(value,type(value))
    # print(number)
    temp = np.zeros(len(value),dtype=np.uint8)
    e = np.copy(value)
    e.sort()
    # print(e)
    for item in e:
        temp[np.argwhere(e==item)] = number[np.argwhere(value==item)]

    # print(temp) 
    # plt.plot(e/10.,temp)
    # plt.show()
    return e,temp

def disd(image,fineBase,pipeMask):
    """
    Return means of circle area of image and fineBase.
    """
    disdent0 = 0
    disdent1 = 0
    for x,y in np.argwhere(pipeMask>0):
        disdent0 += image[x,y]
        disdent1 += fineBase[x,y]

    return disdent0/len(np.argwhere(pipeMask>0)), disdent1/len(np.argwhere(pipeMask>0))
        
# # load fineBase
# fineBase = plt.imread("fineBase_gp0.bmp")
# load pipeMask
pipeMask = plt.imread("pipeMask.bmp")

# kernel for threshold
kernel = np.ones((3,3),dtype=np.uint8)
kernel[0,0] = 0
kernel[0,2] = 0
kernel[2,0] = 0
kernel[2,2] = 0
# kernel[1,1] = 0

d_list = [] # ???

cf =open("%s/seprate_mask_gp.csv" %savePath,"w")
writer = csv.writer(cf)
for d in os.listdir(loadPath):
    fineBase = np.zeros((96,96),dtype=float)

    for item in os.listdir("%s/%s" %(loadPath,d)):
        img = cv2.resize(plt.imread("%s/%s/%s" %(loadPath,d,item)),(96,96),interpolation=cv2.INTER_AREA)
        fineBase = fineBase + img

    fineBase /= len(os.listdir("%s/%s" %(loadPath,d)))

    # d_iter = 0
    
    for fn in os.listdir("%s/%s" %(loadPath,d)):
        image = cv2.resize(plt.imread("%s/%s/%s" %(loadPath,d,fn)),(96,96),interpolation=cv2.INTER_AREA)
        writer.writerow([fn,d])
        # d0,d1 = disd(image,fineBase,pipeMask)
        # fineBase = fineBase -d1 +d0

        # image = cv2.resize(image,(294,294),interpolation=cv2.INTER_AREA)
        # f = np.fft.fft2(image/fineBase)
        # fshift = np.fft.fftshift(f)
        # ms = 20*np.log(np.abs(fshift))
        # _,d_mat = value(96,ms,220)

        # f_ishift = np.fft.ifftshift(fshift)
        # img_b = np.abs(np.fft.ifft2(f_ishift))

        # th,mst = bAndN(ms,image)

        # plt.imshow(ms)
        # plt.title(fn)
        # plt.colorbar()
        # plt.savefig("%s/bp%s/%s.png" %(savePath,d,fn[:-4]))
        # plt.close()
        # maxv = np.max(fshift)
        # minv = np.min(fshift)

        # fshift = (fshift-maxv)/(maxv-minv)
        # cv2.imwrite("%s_mask/bp%s/%s.bmp" %(savePath,d,fn[:-4]),image/fineBase)

        # plt.imshow(image/fineBase)
        # plt.title(fn)
        # plt.colorbar()
        # plt.savefig("%s_mask/bp%s/%s.png" %(savePath,d,fn[:-4]))
        # plt.close()

        # plt.imshow(th)
        # plt.title(fn)
        # plt.colorbar()
        # plt.savefig("%s_mask_th/bp%s/%s.png" %(savePath,d,fn[:-4]))
        # plt.close()
        
        # plt.imshow(mst)
        # plt.title(fn)
        # plt.colorbar()
        # plt.savefig("%s_mask_nm/bp%s/%s.png" %(savePath,d,fn[:-4]))
        # plt.close()

        # mx = cv2.morphologyEx(th,op=cv2.MORPH_OPEN,kernel=kernel)
        # plt.imshow(mx)
        # plt.title(fn)
        # plt.colorbar()
        # plt.savefig("%s_mask_th_morphOp/bp%s/%s.png" %(savePath,d,fn[:-4]))
        # plt.close()

        dm = seprate(image/fineBase,pipeMask)
        e,t = dist(image/fineBase,pipeMask)

        writer.writerow(e)
        writer.writerow(t)

        # plt.plot(0.1*e,t,color=d_iter)

        # plt.imshow(dm)
        plt.plot(e*0.001,t)
        plt.title(fn)
        # plt.colorbar()
        plt.savefig("%s/gp%s/%s.png" %(savePath,d,fn[:-4]))
        plt.close()
        
        # e,t = dist(image/fineBase,pipeMask)

        # plt.plot(e,t)
        # plt.show()

        # cv2.imwrite("%s_base/gp%s/%s" %(savePath,d,fn),image/dm)

cf.close()
