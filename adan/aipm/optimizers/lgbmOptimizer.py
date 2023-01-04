import numpy as np
from xgboost.sklearn import XGBRegressor, XGBClassifier
from collections import OrderedDict
from adan.aipm.optimizers.optimizer_core import *
from adan.metrics.metrics_utilities import regressionMix,classificationMix
import xgboost as xgb
import scipy
from lightgbm import LGBMClassifier,LGBMRegressor


class lightGBMOptimizer(Optimizer):
    def __init__(self,train,target,task,ratio_evals=20,metric=[],n_jobs=1,randomize=True,boosting='gbdt'):
        self.ratio_evals=ratio_evals
        self.n_jobs=n_jobs
        if task=="regression":
            self.model=LGBMRegressor(nthread=n_jobs,boosting=boosting)
        elif task=="classification":
            self.model=LGBMClassifier(nthread=n_jobs,boosting=boostinh)
        
        n_features=train.shape[1]*1.0


            
        n_estimators=scipy.stats.randint(10,max(train.shape[1],30))
        max_depth=scipy.stats.randint(1,min(train.shape[1],30))
        lambda_l1=np.arange(0,0.8,0.5/ratio_evals)
        lambda_l2=np.arange(0,0.8,0.5/ratio_evals)
        extra_trees=[True,False]
        subsample=np.arange(0.2,1.025,0.25/ratio_evals)
        bagging_freq=scipy.stats.randint(1,200)
        colsample_bytree=np.arange(0.3,1.025,0.25/ratio_evals)
        num_leaves=scipy.stats.randint(1,100)
        drop_rate=np.arange(0.1,0.8,0.25/ratio_evals)
        skip_drop=np.arange(0,0.8,0.25/ratio_evals)
        alpha=scipy.stats.randint(10,2000)
        learning_rate=[0.00001,0.0001,0.001,0.01,0.015,0.02]+np.arange(0.02,0.555,0.005).tolist()
    
    
        grid = {'n_estimators':n_estimators,'max_depth':max_depth, 'subsample':subsample,
                  'colsample_bytree':colsample_bytree,
                  'learning_rate':learning_rate,
                  'num_leaves':num_leaves,
                  'extra_trees':extra_trees,
                  'lambda_l1':lambda_l1,
                  'lambda_l2':lambda_l2,
                  'bagging_freq':bagging_freq,
                  'drop_rate':drop_rate,
                  'alpha':alpha}
    
        grid=OrderedDict(grid)
        
        self.types={'n_estimators':int,'max_depth':int,'num_leaves':int,'extra_trees':bool,'bagging_freq':int}
        self.constraints={'n_estimators':(1,n_features),'bagging_freq':(0,50),'drop_rate':(0,0.5),'extra_trees':(False,True),
                          'alpha':(0,2),'lambda_l1':(0,2),'lambda_l2':(0,2),'num_leaves':(10,10000),
                          'learning_rate':(0.001,1.0),'max_depth':(1,n_features),'colsample_bytree':(0.1,0.5),'subsample':(0.1,0.5)}
        
        super(lightGBMOptimizer,self).__init__(model=self.model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
            randomize=randomize,metric=metric)
            

        
