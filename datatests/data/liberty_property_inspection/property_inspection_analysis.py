# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:56:31 2016

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
from adan.features.utilities import *
from adan.genetics.genetic_programming import *
from adan.modelling.estimation_utilities import *
from adan.modelling.symbolic_modelling import *
from adan.modelling.optimizers import *
from adan.modelling.metrics.gini import *


target_name='Hazard'

ngen=500
df=readDataCSV(os.path.dirname(os.path.abspath(__file__))+"/train.csv")
df_processed,target_processed,scaler=prepareData(df,target_name)
df2,target=sample(df_processed,target_processed,fraction=0.1)
#df=df.sample(n=100)

df2=chooseQuantileBest(df2,target,limit_n=5000,quant=0.5,processes=6)[0]
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=30,population=1000,features=500,evaluator=evalPearsonCorNumba,task="regression",n_processes=2)
#res=regression(np.column_stack(g['best_features_plus_cols']),target,n_iter_search=5,n_jobs=4)
res=xgbOptimizerTree(np.column_stack(g['best_features_plus_cols']),target,max_evals=10,nfolds=3,ratio_evals=15,max_evals_optim=10,n_jobs=6,randomize=True,metric=normalized_gini)
#res=xgbOptimizerTree(df2,target,max_evals=10,nfolds=3,ratio_evals=15,max_evals_optim=2,n_jobs=6,randomize=True,metric=normalized_gini)

model=res[1]

newinput=calcNewFeatures(g,df_processed)
newinput=df_processed
model.fit(np.column_stack(newinput[0]),target_processed)
df_test=readData(os.path.dirname(os.path.abspath(__file__))+"/test.csv")

idvar=df_test.ix[:,'Id']
df_test=prepareData(df_test,target_name=None,scaler=scaler)[0]
test_input=np.column_stack(calcNewFeatures(g,df_test)[0])

predictions=[]
for idv,row in zip(idvar,test_input):
    predictions.append([idv,model.predict([row])[0]])

predictions=pd.DataFrame(predictions,columns=['Id','Hazard'])
predictions.to_csv('results.csv',index=False)
print(res)