from collections import OrderedDict
from .optimizer_core import *
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn import decomposition

class pcaElasticNetOptimizer(Optimizer):
    def __init__(self,train,target,task,ratio_evals=10,metric=[],tolerance=-1,n_jobs=1,randomize=False):

        if task=="regression":
            model=Pipeline(steps=[('pca', decomposition.PCA()), ('elastic', linear_model.ElasticNet())])
        elif task=="classification":
            model=Pipeline(steps=[('pca', decomposition.PCA()), ('elastic', linear_model.SGDClassifier(penalty="elasticnet",loss="log"))])
        
        n_features=train.shape[1]*1.0
        pcrange=np.arange(np.round(train.shape[1]/4.0),n_features,int(n_features/ratio_evals))    
        pcrange=[int(k) for k in pcrange]
    
        ratio_range=np.arange(0.01,1.0,1.0/ratio_evals)
    
        grid={'pca__n_components':pcrange,'elastic__l1_ratio':ratio_range}
        grid=OrderedDict(grid)
        self.types={'pca__n_components':int}  
        self.n_jobs=n_jobs
        
        self.constraints={'pca__n_components':(1,n_features),'elastic__l1_ratio':(0.01,1.0)}
    
        super(pcaElasticNetOptimizer,self).__init__(model=model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,tolerance=tolerance,
                randomize=randomize,metric=metric)



class elasticNetOptimizer(Optimizer):
    def __init__(self,train,target,task,ratio_evals=10,metric=[],tolerance=-1,n_jobs=1,randomize=False):

        if task=="regression":
            model=linear_model.ElasticNet()
        elif task=="classification":
            model=linear_model.SGDClassifier(penalty="elasticnet",loss="log")
        
        ratio_range=np.arange(0.0,1.0,1.0/ratio_evals)
                    
        grid={'l1_ratio':ratio_range}
        grid=OrderedDict(grid)
        
        self.constraints={'l1_ratio':(0.01,1.0)}
        self.types={}
        self.n_jobs=n_jobs
    
        super(elasticNetOptimizer, self).__init__(model=model,task=task,param_grid=grid,types=self.types,train=train,target=target,constraints=self.constraints,
                randomize=randomize,metric=metric)
