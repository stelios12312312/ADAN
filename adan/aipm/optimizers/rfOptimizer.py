from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
from . import optimizer_core
from .optimizer_core import Optimizer
from collections import OrderedDict
from adan.metrics.metrics_utilities import *

class randomForestOptimizer(Optimizer):
    def __init__(self,train,target,task,ratio_evals=10,metric=[],randomize=True,n_jobs=1):
        if task=="regression":
            self.model=RandomForestRegressor(warm_start=False, oob_score=True)
        elif task=="classification":
            self.model=RandomForestClassifier(warm_start=False, oob_score=True)
        else:
            raise Exception('no task specified!')
        #else model=model which uses n_features
        n_features=train.shape[1]*1.0
        dummy=int(n_features**1.75)
        if dummy<100:
            dummy=100
            
        self.n_jobs=n_jobs

        n_estimators=np.arange(max(np.round(train.shape[1]/4.0),1),min(5000,dummy),max(int(n_features/ratio_evals),1))

    
        grid={'n_estimators':n_estimators,'max_features':np.arange(1,int(n_features/1.5))}
        grid=OrderedDict(grid)
        
        self.types={'n_estimators':int,'max_features':int}

        self.constraints={'n_estimators':(20,n_features),'max_features': (1,n_features)}
        
        super(randomForestOptimizer,self).__init__(model=self.model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
            randomize=randomize,metric=metric)



    def optimizeModelProtocol(self,test_input=[],test_targets=[],valid_input=[],valid_targets=[],
                 metric=[],activation='linear',n_classes=1,validation_split=0.2,tolerance=0.05,verbose=False,probability=False,n_folds=3):

        self.model.warm_start=True
        
        tolerance=0.001
            
        min_estimators = self.train.shape[1]+10
        max_estimators = (self.train.shape[1]+10)*12
        
        score_new=tolerance
        score_old=-1000000000
        
        estims=np.arange(min_estimators,max_estimators,int(max_estimators/(10.0)))
                
        best=-1
        for n_estims in estims:
            estimators=int(n_estims)
            print("trying forest with n_estimators: "+str(estimators))
            self.model.set_params(n_estimators=estimators)
            self.model.fit(self.train, self.target)
    
            # Record the OOB error for each `n_estimators=i` setting.      
                  
            score_new = max([self.model.oob_score_,0.0000000001])
            
            if verbose:
                print("new score is:"+str(score_new))
                
            if score_new-score_old < tolerance:
                break
            best_estimators=estimators
            score_old=score_new
        
        model_sqrt=RandomForestRegressor(warm_start=True, oob_score=True,max_features='sqrt')
        model_sqrt.set_params(n_estimators=best_estimators)
        
        model_log=RandomForestRegressor(warm_start=True, oob_score=True,max_features='log2')
        model_log.set_params(n_estimators=best_estimators)  
        
        model_sqrt.fit(self.train, self.target)
        model_log.fit(self.train,self.target)
        #note: the oob score is the R^2 for random forests in scikit learn.
        scores=np.array([self.model.oob_score_, model_sqrt.oob_score_, model_log.oob_score_])
        best=np.where(scores==scores.max())[0]
        
        if best==1:
            self.model=model_sqrt
        elif best==2:
            self.model=model_log
            
        if self.task=="regression":
            m=calcMetricsRegression(self.model,self.train,self.target,n_folds=n_folds)
        elif self.task=="classification":
            m=calcMetricsClassification(self.model,self.train,self.target,n_folds=n_folds)
            
        return {'model':self.model,'metrics':m,'score':self.model.oob_score_}
