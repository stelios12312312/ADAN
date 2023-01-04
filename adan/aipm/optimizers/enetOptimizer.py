import numpy as np
from collections import OrderedDict
from optimizers import *
from adan.metrics.metrics_utilities import regressionMix,classificationMix
from sklearn import linear_model

class enetOptim(Optimizer):
    def __init__(self,train,target,task,ratio_evals=10,metric=[],tolerance=-1,n_jobs=1,randomize=False):
        self.ratio_evals=ratio_evals

        if task=="regression":
            model=linear_model.ElasticNet()
        elif task=="classification":
            model=linear_model.SGDClassifier()

        alpha=[0.05,0.2,0.5,0.8,0.1]
        l1_ratio=np.arange(0.0,1.0,1.0/ratio_evals)


        grid={'alpha':alpha,'l1_ratio':l1_ratio}
        grid=OrderedDict(grid)

        self.types={}
        self.constraints={'alpha':(0.01,1.0),'l1_ratio':(0.0,1.0)}

        super(enetOptim,self).__init__(model=model,task=task,param_grid=grid,types=self.types,train=train,target=target,lb=lb,ub=ub,tolerance=tolerance,
            n_jobs=n_jobs,randomize=randomize,metric=metric)