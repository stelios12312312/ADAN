# -*- coding: utf-8 -*-
from adan.metrics.regression import *
from adan.metrics.classification import *
from sklearn.model_selection import ParameterGrid
from adan.aipm.optimizers.optimizers_helpers import *
from adan.aipm.optimizers.ga_hyperparam import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers import hyperparam_optimization
from adan.aipm.optimizers.hyperparam_optimization import gridSearch,runModel
from typing import Dict,List,Callable
import abc
import pandas as pd

class Optimizer(object):
    """
    Core class for optimisation. All model-specific optimisation protcols must implement it.
    """
    
    def __init__(self,model,param_grid:Dict,train,target,task:str,constraints:Dict={},types={},randomize=True,metric:Callable=None):
        """
        

        Parameters
        ----------
        model : TYPE
            DESCRIPTION.
        param_grid : Dict
            Parameter grid (e.g. like in grid search by sklearn).
        train : TYPE
            DESCRIPTION.
        target : TYPE
            DESCRIPTION.
        task : str
            Regression or classification.
        constraints : dictionary, optional
            Dictionary of the format {'parameter':(lower_bound,upper_bound). The default is {}.
        types : dictionary, optional
            Type of each parameter, corresponding to the constraints. E.g. {'n_trees':int}. The default is {}.
        randomize : bool, optional
            If True, the grid will be randomised. The default is True.
        metric : Callable, optional
            The type of metric to use. If None it defaults to a mix of metrics for either 
            regression or classification (depending on the task)

        Returns
        -------
        None.

        """
        self.model=model    
        self.types=types
        self.train=train
        self.target=target
        self.randomize=randomize
        self.task=task
        self.constraints=constraints
        self.types=types
        self.param_grid = param_grid

        if task == "regression":
            if metric is None:
                metric = regressionMix
        elif task == "classification":
            if metric is None:
                metric = classificationMix

        self.metric=metric

    def get_params(self,deep=False):
        return self.model.get_params(deep)

    def fit(self,X,y):
        self.model.fit(X,y)
        
    def predict(self,X):
        return self.model.predict(X)
    
    def predict_proba(self,X):
        return self.model.predict_proba(X)

    @abc.abstractmethod
    def optimizeModelProtocol(self,test_input=[],test_targets=[],valid_input=[],valid_targets=[],
                 metric=[],activation='linear',n_classes=1,validation_split=0.2,task="regression",tolerance=0.05,verbose=False):
        
        """Implement a data science protocol to optimize the model"""
        
        return "protocol not implemented"

    def _parse_population(self,population):
        points=[]
        results=[]
        for params,value in population:
            points.append(params.values())
            results.append(value)
        return points,results


    def optimizeModelHyperparam(self,max_evals:int=10,population:int=100,n_folds:int=3,
                                randomize:bool=True,
                                initial_dataset:List=None,tolerance:float=0.01):       
        
        """
        max_evals: maximum number of evaluations
        
        n_folds: The number of folds for cross-validation
        
        initial_dataset: This is an list of values in the form [(param,value)] which can be used as a starting point
        
        randomize (bool): Whether to randomise the list of parameters or not.
        
        population (int): The population for the genetic algorithm
        
        tolerance (float): this parameter is used by the hyperparameter optimization. If the difference is less than tolerance for successive
        iterations, then we stop.
        
        Returns: a dictionary with metrics, best model and the parameters
        
        """
        
        #if there is no initial set of points, then create a grid, and the perform grid search to set up the initial dataset
        if initial_dataset is None:
            points,results=self._gridSearch(max_evals=max_evals,n_folds=n_folds)
        else:         
            points,results=self._parse_population(initial_dataset)
                                  
        final_params=hyperparam_optimization.hyperOptim(model=self.model,points=points,results=results,train=self.train,
                                target=self.target,param_grid=self.param_grid,
                              max_evals=max_evals,constraints=self.constraints,tolerance=tolerance,
                              n_folds=n_folds,types=self.types,
                              metric=self.metric,population=population)

                              
        self.model.set_params(**final_params)

        if self.task=="regression":
            m=calcMetricsRegression(self.model,self.train,self.target,n_folds=n_folds)
        elif self.task=="classification":
            m=calcMetricsClassification(self.model,self.train,self.target,n_folds=n_folds)
    
        return {'metrics':m,'model':self.model,'parameters':points}


    def _comparePerformance(self,perfs,tolerance):
        perfs.sort()
        best=perfs[(len(perfs)-1)]
        second_best=perfs[(len(perfs)-2)]
        if (best-second_best)/second_best < tolerance:
            #True in this case means that performance is not increasing
            return True
        else:
            return False


    def _create_sklearn_scorer(self,metric,probability=False):
        """
        the EvolutionaryAlgorithmSearchCV requires a scorer object, so we have to create a function to feed it.
        The scorer has a signature (clf,X,y)
        :param metric:
        :return:
        """

        def scorer(clf,X,y):
            if probability:
                res=clf.predict_proba(X)
            else:
                res=clf.predict(X)
            error=metric(y,res)
            return error

        return scorer
        

    def optimizeModelGA(self,population:int=10,tournament_size:int=3,mutation_prob:float=0.2,generations:int=20,
                        verbose:bool=True,
                        n_folds:int=3,fit_params=None,n_jobs:int=1,probability=False,**kwargs):

        cv = EvolutionaryAlgorithmSearchCV(estimator=self.model,
                                           params=self.param_grid,
                                           scoring=self._create_sklearn_scorer(self.metric,probability),
                                           cv=KFold(n_splits=n_folds,shuffle=True),
                                           verbose=verbose,
                                           population_size=population,
                                           gene_mutation_prob=mutation_prob,
                                           tournament_size=tournament_size,
                                           generations_number=generations,
                                           refit=True,
                                           gene_type=self.types,
                                           maximize=True,fit_params=fit_params,n_jobs=n_jobs, error_score=0)
        cv.fit(self.train.values, self.target)

        self.model=cv.best_estimator_

        if self.task=="regression":           
            m=calcMetricsRegression(cv.best_estimator_,self.train,self.target,n_folds=n_folds)
        elif self.task=="classification":
            m=calcMetricsClassification(cv.best_estimator_,self.train,self.target,n_folds=n_folds)
            
        return {'metrics':m,'model':cv.best_estimator_,'population':cv.population_,'estimators':cv.population_estimators}
        
               
        
    def _gridSearch(self,max_evals,n_folds):

        if self.randomize:
            grid=ParameterSampler(self.param_grid,n_iter=max_evals)
        else:
            grid=ParameterGrid(self.param_grid)
        
        points=[]
        results=[]
        evaluation=0
        
        for setting in grid:
            if max_evals>0 and evaluation>max_evals:
                print('maximum number of evaluations reached for grid search')
                break
            print('evaluating...')
            print(setting)
            new_points,new_results=runModel(model=self.model,params=setting,
                                            train=self.train,target=self.target,
                                            n_folds=n_folds,n_jobs=self.n_jobs,
                                            types=self.types,metric=self.metric)
            points.append(new_points)
            results.append(new_results)
            evaluation+=1
            
            print(evaluation)
        points=pd.DataFrame(points)
        return points,results
    
        
        
        




    
