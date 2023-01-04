import numpy as np
from xgboost.sklearn import XGBRegressor, XGBClassifier
from collections import OrderedDict
from adan.aipm.optimizers.optimizer_core import *
from adan.metrics.metrics_utilities import regressionMix,classificationMix
import xgboost as xgb
import scipy

class xgbOptimTree(Optimizer):
    def __init__(self,train,target,task,ratio_evals=20,metric=[],n_jobs=1,randomize=False,objective="reg:linear"):
        self.ratio_evals=ratio_evals
        self.n_jobs=n_jobs
        if task=="regression":
            self.model=XGBRegressor(nthread=n_jobs,objective=objective)
        elif task=="classification":
            self.model=XGBClassifier(nthread=n_jobs,objective=objective)
        
        n_features=train.shape[1]*1.0
        self.ratio=ratio_evals

            
        n_estimators=scipy.stats.randint(10,max(train.shape[1],30))
        max_depth=scipy.stats.randint(1,min(train.shape[1],30))
        lambda_l1=np.arange(0,0.8,0.5/ratio_evals)
        lambda_l2=np.arange(0,0.8,0.5/ratio_evals)
        extra_trees=[True,False]
        subsample=np.arange(0.2,1.025,0.25/ratio_evals)
        bagging_freq=scipy.stats.randint(1,200)
        colsample_bytree=np.arange(0.3,1.025,0.25/ratio_evals)
        num_leaves=scipy.stats.randint(1,100)
        boosting=['gbdt','dart','rf']
        drop_rate=np.arange(0.1,0.8,0.25/ratio_evals)
        skip_drop=np.arange(0,0.8,0.25/ratio_evals)
        alpha=scipy.stats.randint(10,2000)
        learning_rate=[0.00001,0.0001,0.001,0.01,0.015,0.02]+np.arange(0.02,0.555,0.005).tolist()
    
    
        grid = {'n_estimators':n_estimators,'max_depth':max_depth, 'subsample':subsample,
                  'colsample_bytree':colsample_bytree,
                  'learning_rate':learning_rate,
                  'num_leaves':num_leaves,
                  'boosting':boosting,
                  'extra_trees':extra_trees,
                  'lambda_l1':lambda_l1,
                  'lambda_l2':lambda_l2,
                  'bagging_freq':bagging_freq,
                  'drop_rate':drop_rate,
                  'alpha':alpha}
    
        grid=OrderedDict(grid)
        
        self.types={'n_estimators':int,'max_depth':int}
        self.constraints={'n_estimators':(1,n_features),'learning_rate':(0.001,1.0),'max_depth':(1,n_features),'colsample_bytree':(0.1,1.0),'subsample':(0.1,1.0)}
        
        super(xgbOptimTree,self).__init__(model=self.model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
            randomize=randomize,metric=metric)
            
    def plot_importance(self):
        xgb.plot_importance(self.model._Booster)
        
#to do
class xgbOptimLinear(Optimizer):
    def __init__(self,train,target,task,ratio_evals=20,metric=[],n_jobs=1,randomize=False,objective="reg:linear"):
        self.ratio_evals=ratio_evals

        if task=="regression":
            self.model=XGBRegressor(nthread=n_jobs,objective=objective)
            self.model.booster='gblinear'
        elif task=="classification":
            self.model=XGBClassifier(nthread=n_jobs,objective=objective)
            self.model.booster='gblinear'
        
        n_features=train.shape[1]*1.0

        #do not let the estimators go below 100
        dummy=int(n_features**1.75)
        if dummy<100:
            dummy=100

        n_estimators=np.arange(max(np.round(train.shape[1]/4.0),1),min(5000,dummy),max(int(n_features/ratio_evals),1))
        learning_rate=np.arange(0.0001,0.5,0.5/ratio_evals)  
        reg_alpha=np.arange(0.1,1.0,1.0/ratio_evals)
        reg_lambda=np.arange(0.1,1.0,1.0/ratio_evals)  
    
        grid={'n_estimators':n_estimators,'learning_rate':learning_rate,'reg_alpha':reg_alpha,'reg_lambda':reg_lambda}
        grid=OrderedDict(grid)
        
        self.types={'n_estimators':int,'max_depth':int}
        self.constraints={'n_estimators':(1,n_features),'learning_rate':(0.001,1.0),'reg_alpha':(0.0,1.0),'reg_lambda':(0.0,1.0)}
        
        super(xgbOptimTree,self).__init__(model=self.model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
            randomize=randomize,metric=metric)
            
    def plot_importance(self):
        xgb.plot_importance(self.model._Booster)