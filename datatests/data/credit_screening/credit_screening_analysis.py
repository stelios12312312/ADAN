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

from adan.modelling.symbolic_fitting import *
import sympy

target_name=15
ngen=3

df=pd.read_csv(os.path.dirname(os.path.abspath(__file__))+"/crx.csv",sep=",",na_values="?",header=None)
df=df.dropna()

#df=df[np.isfinite(df[target_name])]

df2,target=prepareData(df,target_name)
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=10,population=300,features=50,evaluator=evalANOVA,task="classification")

clf=sklearn.linear_model.LogisticRegression()

res=cross_val_predict(clf,np.column_stack(g['best_features'][0:20]),target)
classify(np.column_stack(g['best_features']),target)
eqs=convertIndividualsToEqs(g['best_individuals'],df2.columns)
print(sympy.simplify(sympy.sympify(eqs)))