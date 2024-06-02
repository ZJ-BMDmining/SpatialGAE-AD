# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 11:37:03 2023

@author: ssshe
"""
from __future__ import division
import pandas as pd
import sys
import time
import os
import scipy
from utils import load_data
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_str',default='ACT',type=str, help='name of dataset, ROSMAP MSBB ACT Mayo ANM1 ANM2')
parser.add_argument('--n_clusters',default=4,type=int, help='expected number of clusters')
parser.add_argument('--k', default=None, type=int, help='number of neighbors to construct the cell graph')
parser.add_argument('--is_NE', default=True, type=bool, help='use NE denoise the cell graph or not')


args = parser.parse_args()
n_clusters = args.n_clusters

if not os.path.exists('result/'):
    os.makedirs('result/')
    
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

# Paths
data_path = './data/'+args.dataset_str+'/'+data_name+'.csv'

print(data_path, data_name, args.is_NE)

A, X, cells = load_data(data_path, args.dataset_str,
                               data_name, args.is_NE, n_clusters, args.k)

print(args.dataset_str+' graph matrix finished')