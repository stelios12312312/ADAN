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
from sklearn.ensemble import GradientBoostingRegressor
import time


target_name=13
ngen=30
df=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/housing.csv",sep=',',header=-1)

df=df[np.isfinite(df[target_name])]


df2,target=prepareData(df,target_name)
t0=time.time()
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=10,population=1000,features=5,task="regression",evaluator=evalPearsonCorNumba,n_processes=1)
t1=time.time()
total=t1-t0
print(total)

#feats=calcMetricParallel(df,target)
#df4=choosePercBest(df3,feats,perc=0.95)

clf=GradientBoostingRegressor(n_estimators=30)
#clf=sklearn.linear_model.LinearRegression()

res=cross_val_predict(clf,np.column_stack(g['best_features']),target)
plt.scatter(target,res)