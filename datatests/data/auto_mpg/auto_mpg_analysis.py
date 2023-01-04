# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:45:47 2016

@author: stelios
"""

# -*- coding: utf-8 -*-
import pandas as pd

import os, sys
lib_path = os.path.abspath(os.path.join('..','..','..'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)


import adan
from adan.features.utilities import *
from adan.genetics.genetic_programming import *
from adan.features.utilities import *
from adan.genetics.genetic_programming import *
from adan.modelling.estimation_utilities import *
from adan.modelling.symbolic_modelling import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from adan.modelling.xgboost_wrapper import *
from adan.modelling.metrics import regression_metrics
import sklearn
from sklearn.metrics import metrics
from sklearn.cross_validation import cross_val_predict
import numpy as np
from sklearn.cross_validation import cross_val_predict
import sklearn.linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor


target_name='V1'
ngen=500
df=readData('auto_mpg.csv')

#df.drop('V9',axis=1,inplace=True)
df2,target,scaler=prepareData(df,target_name)
df2=chooseQuantileBest(df2,target,limit_n=1000,perc=0.9)[0]

g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=3,score_weight=10,population=2000,features=5)
res=regression(np.column_stack(g['best_features'][0:20]),target)
k=findSymbolicExpressionL1(df2,target,g,scaler,task='regression')