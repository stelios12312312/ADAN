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
from adan.features.feature_selection import *

#from sklearn import svm
#from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from adan.modelling.estimation_utilities import *



target_name='count'
ngen=3

df=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/train.csv",sep=",",na_values="?",header=0)
df2,target=prepareData(df,target_name)

dummy=choosePercBest(df2,target,perc=0.75)

df2=dummy[0]

g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=10,population=3000,features=50,task="regression")


scores=regression(np.column_stack(g['best_features'][0:20]),target)

#do feature selection with mic()

#FIX THE ISSUE WITH THE HARMONIC MEAN