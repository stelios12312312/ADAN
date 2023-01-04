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
from adan.modelling.estimation_utilities import *
from adan.modelling.symbolic_conversion import *
from sympy import *
from adan.modelling.symbolic_modelling import *

target_name='class'
ngen=10

df=readData(os.path.dirname(os.path.abspath(__file__))+"/breast_cancer.csv")

df2,target,scaler=prepareData(df,target_name)
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=5,score_weight=5,population=4000,features=10,evaluator=evalANOVANumba,task="classification")
calcMICReg(df2,target,df2.columns[0])

#res=classify(np.column_stack(g['best_features']),target)
eqs=convertIndividualsToEqs(g['best_individuals'],df2.columns)

k=findSymbolicExpression(df2,target,g,scaler,task="classification")
