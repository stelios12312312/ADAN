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

#from sklearn import svm
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from adan.modelling.estimation_utilities import *


target_name='Class'
ngen=3

df=readData(os.path.dirname(os.path.abspath(__file__))+"/arcene.csv")


df=df[np.isfinite(df[target_name])]

df2,target=prepareData(df,target_name)
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=10,population=3000,features=50,evaluator=evalANOVA,task="classification")


#feats=calcMetricParallel(df,target)
#df4=choosePercBest(df3,feats,perc=0.95)

#clf=GradientBoostingRegressor(n_estimators=500)
clf=sklearn.linear_model.LogisticRegression()

res=classify(np.column_stack(g['best_features'][0:20]),target)
