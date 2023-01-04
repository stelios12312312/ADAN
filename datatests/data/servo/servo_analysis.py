# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:45:47 2016

@author: stelios
"""

# -*- coding: utf-8 -*-
import pandas as pd

import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)


from adan.features.utilities import *
from adan.genetics.genetic_programming import *


import numpy as np
from sklearn.cross_validation import cross_val_predict
import sklearn.linear_model
import matplotlib.pyplot as plt


target_name=4
ngen=3
df=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/servo.csv",sep=',',header=-1)

df=df[np.isfinite(df[target_name])]

df2,target=prepareData(df,target_name)
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=10,population=2000,features=10)


#feats=calcMetricParallel(df,target)
#df4=choosePercBest(df3,feats,perc=0.95)

#clf=GradientBoostingRegressor(n_estimators=30)
clf=sklearn.linear_model.LinearRegression()

res=cross_val_predict(clf,np.column_stack(g['best_features']),target)
plt.scatter(target,res)