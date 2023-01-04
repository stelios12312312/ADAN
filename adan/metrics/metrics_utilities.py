import numpy as np

#from sklearn.cross_validation import cross_val_predict
from sklearn.model_selection import *
from adan.metrics.regression import *
from adan.metrics.classification import *
from adan.metrics.metrics_utilities import *
#from sklearn.grid_search import ParameterGrid
#from sklearn.grid_search import ParameterSampler
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import sklearn


def scorer_safe(estimator,X,y):
    """
    scorer_safe is used because if there is a nan or inf value, then cross_validation by scikit-learn breaks down.
    """
    res=estimator.predict(X)
    if np.logical_not(all(np.isfinite(res))):
        return -2.0
    return corrNumba(res,y)

def classificationMix(target,res):
    """
    this is a combined classification metric used as the heuristic for GA optimization in classification
    :param target:
    :param res:
    :return:
    """
    acc=metrics.accuracy_score(target,res)
    f1=metrics.f1_score(target,res,average='macro')
    kappa=metrics.cohen_kappa_score(target,res)
    mix=acc*f1*kappa
    return mix

def regressionMix(target,res):
    """
    this is a combined classification metric used as the heuristic for GA optimization in classification
    :param target:
    :param res:
    :return:
    """
    cor = pearson_correlation_coefficient(target, res)
    concor = concordance_correlation_coefficient(target, res)
    r2=metrics.r2_score(target,res)
    mix = np.mean([concor, cor, r2])
    return mix

def calcMetricsRegression(model,train,target,n_folds=5):
    res=sklearn.model_selection.cross_val_predict(estimator=model,X=train,y=target,cv=n_folds,n_jobs=1)
    
    cor=pearson_correlation_coefficient(res,target)
    concor=concordance_correlation_coefficient(res,target)
    mae=mean_absolute_error(res,target)
    mse=mean_squared_error(res,target)
    medae=median_absolute_error(res,target)
    r2=metrics.r2_score(target,res)
    mix=regressionMix(target,res)

    scores={'cor':cor,'concor':concor,'mae':mae,'mse':mse,'medae':medae,'r2':r2,'mix':mix}

    return scores



def calcMetricsClassification(model,train,target,n_folds=5):
    res=sklearn.model_selection.cross_val_predict(estimator=model,X=train,y=target,cv=n_folds,n_jobs=1)

    acc=metrics.accuracy_score(target,res)
    f1=metrics.f1_score(target,res,average='macro')
    kappa=metrics.cohen_kappa_score(target,res)
    mix=classificationMix(target,res)

    scores={'acc':acc,'f1':f1,'kappa':kappa,'mix':mix}

    return scores