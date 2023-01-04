# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:56:31 2016

@author: stelios
"""

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
from adan.features.utilities import *
from adan.genetics.genetic_programming import *
from adan.modelling.estimation_utilities import *
from adan.modelling.symbolic_modelling import *
from adan.modelling.optimizers import *
from adan.modelling.metrics.gini import *


target_name='Hazard'

ngen=3
df=readData(os.path.dirname(os.path.abspath(__file__))+"/train.csv")
df_processed,target_processed,scaler=prepareData(df,target_name,del_zero_var=True)
df2,target=sample(df_processed,target_processed,fraction=0.55)
print("shape of data is: "+df2.shape)
#df=df.sample(n=100)

#df2=chooseQuantileBest(df2,target,limit_n=5000,quant=0,processes=6)[0]
g=findFeaturesGP(df=df2,targets=target,ngen=ngen,max_tree=30,population=1500,features=500,evaluator=evalPearsonCorNumba,task="regression",n_processes=2)
#res=regression(np.column_stack(g['best_features_plus_cols']),target,n_iter_search=5,n_jobs=4)
res=randomForestOptimizer(np.column_stack(g['best_features_plus_cols']),target,ratio_evals=5,nfolds=5)
#res=randomForestOptimizer(np.column_stack(g['all_features']),target,ratio_evals=20,nfolds=5)
#res=randomForestOptimizer(df2,target,ratio_evals=20,nfolds=5)

model=res[1]
newinput=np.column_stack(calcNewFeatures(g,df_processed,'all')[0])
#newinput=df_processed.values

model.fit(newinput,target_processed)
df_test=readData(os.path.dirname(os.path.abspath(__file__))+"/test.csv")

idvar=df_test.ix[:,'Id']
df_test=prepareData(df_test,target_name=None,scaler=scaler,del_zero_var=False,match_columns=df2.columns)[0]
test_input=np.column_stack(calcNewFeatures(g,df_test)[0])
#test_input=df_test.values

predictions=[]
for idv,row in zip(idvar,test_input):
    predictions.append([idv,model.predict([row])[0]])

predictions=pd.DataFrame(predictions,columns=['Id','Hazard'])
predictions.to_csv('results.csv',index=False)
print(res)