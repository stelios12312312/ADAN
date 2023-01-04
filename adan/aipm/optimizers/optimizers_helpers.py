
import numpy as np

#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import *
from adan.metrics.regression import *
from adan.metrics.regression import *
from adan.metrics.classification import *
#from sklearn.grid_search import ParameterGrid
#from sklearn.grid_search import ParameterSampler
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import sklearn


def gridSearch(model,param_grid,train,target,max_evals=-1,ratio_evals=5,n_jobs=2,nfolds=5,types={},randomize=True,metric=corrNumba):    
    
    if randomize:
        grid=sklearn.model_selection.ParameterSampler(param_grid,n_iter=max_evals)
    else:
        grid=sklearn.model_selection.ParameterGrid(param_grid)
    
    points=[]
    results=[]
    evaluation=0
    
    for setting in grid:
        if max_evals>0 and evaluation>max_evals:
            print('maximum number of evaluations reached for grid search')
            break
        print(setting)
        new_points,new_results=runModel(model=model,settings=setting,train=train,target=target,nfolds=nfolds,n_jobs=n_jobs,types=types,metric=metric)
        points.append(new_points)
        results.append(new_results)
        evaluation+=1
        
        print(evaluation)
        
    return np.array(points),np.array(results)


def runModel(model,params,train,target,n_jobs=1,n_folds=3,types={},metric=corrNumba):
    results=[]
    try:
        params=params.iloc[0,:].to_dict()
    except:
        pass

    if len(types)>0:
        for t,fun in types.items():
            params[t]=fun(params[t])
        
    crossval_res=sklearn.model_selection.cross_val_predict(estimator=model,X=train,y=target,cv=n_folds,n_jobs=n_jobs)
    if all(np.isfinite(crossval_res)):
        #this covers the case were the output comes in the form of [[1],[2],[3],...]
        if(len(crossval_res.shape)>1):
            if crossval_res[0].shape==1:
                crossval_res=np.ndarray.flatten(crossval_res)
        results=metric(target,crossval_res)
    else:
        results=0.0
        
    return params,results