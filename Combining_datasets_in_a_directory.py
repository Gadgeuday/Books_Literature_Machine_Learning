# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 11:29:00 2023

@author: gadge
"""

import os
import pandas as pd

path = 'C:/Users/gadge/OneDrive/Desktop/MachineLearning/Project/Data'
filepath = path+'/NYTimes_Bestsellers'
filelist = os.listdir(filepath)

datasets = []

for file in filelist:
    df = pd.read_csv(filepath+'/'+file)
    datasets.append(df)
    

df_full = pd.concat(datasets)
df_full.to_csv(path+'/NewYorkTimesBestsellersLists.csv',index=False)

