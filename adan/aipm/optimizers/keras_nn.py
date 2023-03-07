from sklearn import decomposition
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping
from adan.metrics.regression import *
from adan.metrics.classification import *
from adan.metrics.metrics_utilities import *
import pandas as pd
from adan.metrics.metrics_utilities import classificationMix,regressionMix
from adan.aipm.optimizers.optimizer_core import Optimizer
import numpy as np
import copy
import abc

sparse=False
n_classes=1

class sequentialModelExtention(object):
    def __init__(self,n_inputs, n_outputs, layers, neurons, dropout, svd):
        self.model = Sequential()
        self.svd=[]

        self.params={'n_inputs':n_inputs,'n_outputs':n_outputs, 'layers':layers, 'neurons':neurons, 'dropout':dropout, 'svd':svd}
        self.set_params(**self.params)

    def __addDropout(self,value):
        self.model.add(Dropout(np.array([value])[0]))
    
    def _addLayer(self,neurons=500,dropout=0.2):
        self.model.add(Dense(int(neurons)))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(float(dropout)))
        self.model.add(PReLU())

    def get_params(self,deep=True):
        res = copy.deepcopy(self.params)
        return res
    
    
    def set_params(self,n_inputs,n_outputs,layers=1,neurons=10,dropout=0.1,svd=-1):
        self.params=params={'n_inputs':n_inputs,'n_outputs':n_outputs,'layers':layers,'neurons':neurons,'dropout':dropout,'svd':svd}

        if params['svd']==-1:
            self.svd=preprocessing.MinMaxScaler()
        if params['svd']>0:
            self.svd=decomposition.TruncatedSVD(n_components=int(params['svd']))
            params['n_inputs']=int(params['svd'])

        self.model = Sequential()
        self.model.add(Dense(int(params['neurons']), input_dim=int(params['n_inputs'])))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(float(params['dropout'])))
        self.model.add(PReLU())

        for i in range(1,(params['layers']+1)):
            self._addLayer(neurons=params['neurons'],dropout=float(params['dropout']))
            
        self.model.add(Dense(int(params['n_outputs'])))
        self.model.add(Activation(self.activation))

        self.model.compile(loss='mse', optimizer="adam")
        
        return self.model
        
        
    def fit(self,data,targets, nb_epoch=20, batch_size=128,shuffle=True,validation_split=0.1,verbose=False):
        training_input=self.svd.fit_transform(data)
        early=EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto')
        self.model.fit(x=training_input,y=targets, epochs=nb_epoch, batch_size=batch_size,shuffle=shuffle,validation_split=validation_split,callbacks=[early],verbose=verbose)
        
        
    def predict(self,data):
        return self.model.predict(self.svd.transform(data))


class sequentialModelExtensionRegressor(sequentialModelExtention):
    def __init__(self,n_inputs,n_outputs,layers=1,neurons=10,dropout=0.1,svd=-1):
        self.activation = "linear"
        self.task="regression"
        super(sequentialModelExtensionRegressor, self).__init__(n_inputs, n_outputs, layers, neurons, dropout, svd)


class sequentialModelExtensionClassifier(sequentialModelExtention):
    def __init__(self,n_inputs,n_outputs,layers=1,neurons=10,dropout=0.1,svd=-1):
        self.activation = "softmax"
        self.task="classification"
        super(sequentialModelExtensionClassifier,self).__init__(n_inputs, n_outputs, layers, neurons, dropout, svd)
        
    def predict_proba(self,data):
        return self.model.predict_proba(self.svd.transform(data))
        

class deepNNOptim(Optimizer):
    def __init__(self,train,target,task,metric=[],ratio_evals=10,n_jobs=1):
        #FIX RATIO EVALS
        n_inputs=train.shape[1]
        if(len(target.shape)==1):
            n_outputs=1
        else:
            n_outputs=target.shape[1]
            
        self.n_jobs=n_jobs

        if task=="regression":
            model=sequentialModelExtensionRegressor(n_inputs=n_inputs,n_outputs=n_outputs)
        elif task=="classification":
            model=sequentialModelExtensionClassifier(n_inputs=n_inputs,n_outputs=n_outputs)

        n_features=train.shape[1]*1.0

        #do not let the estimators go below 100

        layers=np.array([1,2])  
        if n_features<10000:              
            neurons=np.arange(n_features,min(10000,n_features**2),max(int(n_features/ratio_evals),1))
        else:
            neurons=np.arange(n_features,n_features*10,max(int(n_features/ratio_evals),1)) 
            
        dropout=np.array([0.1,0.2,0.3,0.4,0.5]).tolist()  
    
        grid={'n_inputs':[n_inputs],'n_outputs':[n_outputs],'layers':layers,'neurons':neurons,'dropout':dropout,'svd':np.arange(-1,n_features/2.0,50)}
        grid=OrderedDict(grid)
        
        self.types={'layers':int,'neurons':int,'svd':int}
        self.constraints={'n_inputs':(n_inputs,n_inputs),'n_outputs':(n_outputs,n_outputs),'layers':(1,3),'neurons':(1,10000),'svd':(-1,n_features),'dropout':(0.0,0.5)}
        
        super(deepNNOptim,self).__init__(model=model,param_grid=grid,types=self.types,constraints=self.constraints,train=train,target=target,
                                        metric=metric,task=task)
    

      #overriden method
    def optimizeModelProtocol(self,task,test_input=[],test_targets=[],valid_input=[],valid_targets=[],
                 metric=[],activation='linear',n_classes=1,validation_split=0.2,tolerance=0.05,epochs=20,verbose=False,probability=False):
        """
        The optimizeModelDeep function implements a data science protocol for deep neural networks. It should be applied only for
        larger numbers of features and rows, since applying deepNNs to smaller datasets is probably not very useful.
        """

        n_inputs = self.train.shape[1]
        if len(self.target.shape) == 1:
            n_outputs = 1
        else:
            n_outputs = target.shape[1]

        if metric==[]:
            metric=self.metric

        settings=[
        [1,20,0.0],
        [1,50,0.05],
        [1,100,0.2],
        [1,300,0.2],
        [1,400,0.25],
        [1,500,0.3],
        [2,150,0.2],
        [2,300,0.2],
        [2,400,0.25],
        [2,500,0.3],
        [2,1200,0.4],
        [2,1500,0.5],
        [2,8000,0.8]
         ]                     

        preps=[]
        if sparse and n_inputs>=200:
            preps.append(100)
            if n_inputs>300:
                preps.append(200)
            if n_inputs>500:
                preps.append(300)
        else:
            preps.append(-1)  
        
        perfs=[]
        result={}
        result['model']=self.model
        result['score']=0.0
        
        current_best=-np.inf
        for s in settings:
            for p in preps:

                setting={}
                setting['svd']=p
                setting['layers']=s[0]
                setting['neurons']=s[1]
                setting['dropout']=s[2]
                setting['n_inputs']=n_inputs
                setting['n_outputs']=n_outputs


                if task == "regression":
                    model = sequentialModelExtensionRegressor(n_inputs=n_inputs, n_outputs=n_outputs)
                elif task == "classification":
                    model = sequentialModelExtensionClassifier(n_inputs=n_inputs, n_outputs=n_outputs)

                model.set_params(**setting)
                model.fit(self.train,self.target, nb_epoch=epochs, batch_size=128,shuffle=True,validation_split=validation_split)

                if len(test_input)>0:
                    dat=model.svd.transform(test_input)
                    outp=test_targets
                else:
                    dat=self.train
                    outp=self.target
                    
                if probability:
                    res=model.predict_proba(dat)
                else:                    
                    res=model.predict(dat)
                    
                score=metric(outp,res)
                
                if verbose:
                    print('The score is:'+str(score))
                    
                result[str(setting)]=score
                perfs.append(score)


                if len(perfs)>5 and self._comparePerformance(perfs,tolerance=tolerance):
                    break

                if score>current_best:
                    result['model']=model
                    result['score']=score
                    current_best=score
        
        return result