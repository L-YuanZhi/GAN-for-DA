import os
import cv2 
import csv
import plotly 
import warnings
import seaborn

import numpy as np 
import pandas as pd # data processing, CSV file I/O
import matplotlib.pyplot as plt

from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")
# plotly.offline.init_notebook_mode(connected=True)

# datas = pd.read_csv("input_tensor/model_21928_bp1_none_96x96_mix.csv")
# datas = pd.read_csv("output/minmax_96x96_mix/gp_value.csv")
# datas = pd.read_csv("4410_value_none.csv")

# datas = pd.read_csv("output/KOC_bp.csv")
datas = pd.read_csv("output/cluster_results/fft/sperate_mask_0126/param_gp.csv")

# data = datas[["min","max","mean","norm_mean"]].iloc[:,:].values
ds = datas[["min","max","mean","std"]].iloc[:,:].values
# ds = datas[["row","col","value"]].iloc[:,:].values

# dss = []

# for item in data:
#     dss.append([item[1]/item[0],item[2]/item[3]])

# ds = pd.DataFrame(dss,columns=["max/min","mean/norm_mean"])

# algorithm = (KMeans(n_clusters=2,init="k-means++",n_init=10,max_iter=300,
# tol=0.001,))

algorithm = (OPTICS(xi=0.05,min_samples=15,min_cluster_size=.5))

# algorithm =(DBSCAN(eps = .1, min_samples= 10))

# algorithm = (GaussianMixture(n_components=2, random_state=0))

algorithm.fit(ds)
labels = algorithm.labels_
# centroids = algorithm.cluster_centers_
# print(centroids)

datas["cluster_label"] = labels

datas.to_csv("output/cluster_results/fft/sperate_mask_0126/param_cluster_gp.csv")
# ds["label"] = labels
# load = "/home/lin/pyProject/pipe-roughness/90/mix/cut"
# load = "output/minmax_96x96_mix/minmax_images"
# save = "output/minmax_96x96_mix/mix_gp"
# with open("output/minmax_96x96_mix/KMeans_gp.csv","w") as cf: 
#     writer = csv.writer(cf)
#     writer.writerow(["file_name","label"])
#     for i in range(len(labels)):
#         writer.writerow([datas["file_name"][i],labels[i]])
#         if not os.path.exists(os.path.join(save,str(labels[i]))):
#             os.mkdir(os.path.join(save,str(labels[i])))

#         if datas["file_name"][i][:2]=="NG":
#             f = "bp"
#         else:
#             f = "gp"
        
        # cv2.imwrite(os.path.join(save,str(labels[i]),datas["file_name"][i]),
        #             plt.imread(os.path.join(load,f,datas["file_name"][i])))
        
    
# trace = plotly.graph_objs.Scatter3d(
#     x = datas["min"],
#     y = datas["max"],
#     z = datas["mean"],
#     # x = ds["max/min"],
#     # y = ds["mean/norm_mean"],
#     mode="markers",
#     marker=dict(
#         color=datas["label"],
#         size = 3,
#         line = dict(
#             color = datas["label"],
#             width = 20
#         ),
#         opacity = 0.8
#     )
# )

# data = [trace]
# layout = plotly.graph_objs.Layout(

#     title = "Clusters",
#     scene = dict(
#         xaxis = dict(title = "max/min"),
#         yaxis = dict(title = "mean/norm_mean")
#         # zaxis = dict(title = "label")
#     )
# )
# fig = plotly.graph_objs.Figure(data = data, layout = layout)
# plotly.offline.iplot(fig)