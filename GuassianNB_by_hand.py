# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 17:13:26 2023

@author: gadge
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

x1 = np.random.normal(1,2,200)
y1 = np.random.normal(3,1,200)
df1 = pd.DataFrame([x1,y1]).T
df1.columns = ['x','y']
df1['label'] = 'A'

x2 = np.random.normal(3,2,100)
y2 = np.random.normal(5,1,100)
df2 = pd.DataFrame([x2,y2]).T
df2.columns = ['x','y']
df2['label'] = 'B'

df = pd.concat([df1,df2])

X = np.array(df.drop('label',axis = 1))
y = df['label']

sns.scatterplot(x = 'x',y = 'y', hue = 'label', data = df)


from sklearn.naive_bayes import GaussianNB
NB= GaussianNB(var_smoothing=0)
NB.fit(X, y)

ypred = NB.predict_proba(X)

mean_x_a = np.mean(df[df['label'] == 'A']['x'])
sd_x_a = np.std(df[df['label'] == 'A']['x'])

mean_x_b = np.mean(df[df['label'] == 'B']['x'])
sd_x_b = np.std(df[df['label'] == 'B']['x'])

mean_y_a = np.mean(df[df['label'] == 'A']['y'])
sd_y_a = np.std(df[df['label'] == 'A']['y'])

mean_y_b = np.mean(df[df['label'] == 'B']['y'])
sd_y_b = np.std(df[df['label'] == 'B']['y'])

def likelihood(x,mean,sd):
    z = (x-mean)/sd
    lh = -0.5 * np.log(2.0 * np.pi * sd*sd)
    lh = lh - 0.5 * z *z
    return lh

df['GuassNB_A'] = ypred[:,0]
df['GuassNB_B'] = ypred[:,1]

df['prob_x_a'] = df['x'].apply(lambda x: likelihood(x, mean_x_a, sd_x_a))
df['prob_y_a'] = df['y'].apply(lambda x: likelihood(x, mean_y_a, sd_y_a))
df['prob_x_b'] = df['x'].apply(lambda x: likelihood(x, mean_x_b, sd_x_b))
df['prob_y_b'] = df['y'].apply(lambda x: likelihood(x, mean_y_b, sd_y_b))

class_a = np.log(200/300)
class_b = np.log(100/300)

df['hand_a'] = df['prob_x_a']+df['prob_y_a']+class_a
df['hand_b'] = df['prob_x_b']+df['prob_y_b']+class_b

df['Tot'] = df['hand_a'].apply(lambda x: math.exp(x))+df['hand_b'].apply(lambda x: math.exp(x))
df['Tot'] = np.log(df['Tot'])
df['hand_a_prob'] = np.exp(df['hand_a']-df['Tot'])
df['hand_b_prob'] = np.exp(df['hand_b']-df['Tot'])

df[['GuassNB_A','GuassNB_B','hand_a_prob','hand_b_prob']]

df['norm_prob_a'] = np.exp(df['hand_a'])
df['norm_prob_b'] = np.exp(df['hand_b'])

df['norm_prob_a'] = df['norm_prob_a']/(df['norm_prob_a']+df['norm_prob_b'])
df['norm_prob_b'] = df['norm_prob_b']/(df['norm_prob_a']+df['norm_prob_b'])
