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

from sklearn import svm
from sklearn.ensemble import GradientBoostingRegressor

ngen=3

target_name="area"
df=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/forestfires.csv")

df2,target=prepareData(df,target_name)
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=5,population=4000,features=10)


#feats=calcMetricParallel(df,target)
#df4=choosePercBest(df3,feats,perc=0.95)

clf=GradientBoostingRegressor(n_estimators=100)
clf=sklearn.linear_model.LinearRegression()

res=cross_val_predict(clf,np.column_stack(g['best_features']),target)
plt.scatter(target,res)
