# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 20:04:55 2023

@author: ssshe
"""
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA
from sklearn import manifold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_str', default="ROSMAP",help='name of dataset, ROSMAP MSBB ACT Mayo')
args = parser.parse_args()

if args.dataset_str == "ROSMAP":
    data_name =  "ROSMAP_4738genes"
elif args.dataset_str == "MSBB":
    data_name = "MSBB_19440genes"
elif args.dataset_str == "ACT":
    data_name = "ACT_17574genes"
elif args.dataset_str == "Mayo":
    data_name = "Mayo_13933genes"
elif args.dataset_str == "ANM1":
    data_name = "ANM1"
elif args.dataset_str == "ANM2":
    data_name = "ANM2"

data_matrix = pd.read_csv("./data/"+args.dataset_str+"/"+data_name+".csv", sep=',',header=0,index_col=0)
adjacency_matrix = pd.read_csv("./data/"+args.dataset_str+"/graphmatrix_"+data_name+".csv", sep=',',header=0)

print(data_matrix)
print(adjacency_matrix)
cca = CCA(n_components=500)
output_features2 = data_matrix.values
if(args.dataset_str == "ANM1" or args.dataset_str == "ANM2"):
    output_features2 = output_features2.T

feature_vector2 = adjacency_matrix.values
output, feature = cca.fit_transform(output_features2, feature_vector2)

df = pd.DataFrame(output)
df.to_csv("./data/"+args.dataset_str+"/"+data_name+"_expression.csv")

df = pd.DataFrame(feature)
df.to_csv("./data/"+args.dataset_str+"/"+data_name+"_graph.csv")

df = pd.DataFrame(output+feature)
df.to_csv("./data/"+args.dataset_str+"/"+data_name+"_plus.csv")

df = pd.DataFrame(np.hstack((output,feature)))
df.to_csv("./data/"+args.dataset_str+"/"+data_name+"_concat.csv")
