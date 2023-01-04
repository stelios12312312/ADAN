# -*- coding: utf-8 -*-
import numpy as np
from deap import algorithms,base,cma,creator,tools
from sklearn.gaussian_process import *
# from pyswarm import pso
from adan.metrics.regression import *
from adan.metrics.classification import *
from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor
from adan.aipm.optimizers import *
from adan.aipm.optimizers.optimizers_helpers import  runModel, gridSearch
import pandas as pd
import sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector

def gprSwarmOptim_helper(train,target,max_evals,lb,ub,swarmsize=100):
    """
    optimizer for hyperOptim. Runs a GP and if it fails (which happens if two points have the same input but differen output) tries a random forest.
    Then estimates the optimum using PSO.
    :param train:
    :param target:
    :param max_evals:
    :param lb:
    :param ub:
    :param swarmsize:
    :return:
    """
    try:
        gp = sklearn.gaussian_process.GaussianProcessRegressor(regr='quadratic', corr='squared_exponential',nugget=1e-2, optimizer='Welch')
        gp.fit(train,target)
    except:
        gp=RandomForestRegressor(n_estimators=(50+2*len(train)))
        gp.fit(train,target)
    
    
    xopt = pso(lambda x:-1.0*gp.predict([x]), lb, ub,swarmsize=swarmsize)
    return (xopt[0],-1*xopt[1])


def rfCMAOptim_helper(train,target,max_evals,lb,ub,population=500,ngen=10):
    """
    optimizer for hyperOptim. Uses a random forest along with CMA.
    https://deap.readthedocs.io/en/master/examples/cmaes.html
    :param train:
    :param target:
    :param max_evals:
    :param lb:
    :param ub:
    :param population:
    :return:
    """
    train=train.values
    N=train.shape[1]
    model=RandomForestRegressor(n_estimators=(40+2*N))
    

    model.fit(train,target)
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=train.shape[0])

    #both feasible and distance directly access lb abd ub
    middle_point = [np.mean(k) for k in zip(lb, ub)]
    def evaluator(individual):
        """Feasability function for the individual. Returns True if feasible False
        otherwise."""
        #its important to set it to >=, <= and not <,>
        if np.all(individual>=lb and individual<=ub):
            return model.predict([individual])
        else:
            return -1.0*np.sqrt(sum((np.array(individual)-middle_point)**2)),
        
    toolbox.register("evaluate",evaluator)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    strategy = cma.Strategy(centroid=middle_point, sigma=5.0, lambda_=population)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)   
    algorithms.eaGenerateUpdate(toolbox, ngen=10, halloffame=hof,stats=stats)
    
    
    return (hof[0],evaluator(hof[0]))

def optimizeModel(model,params,train,target,lb,ub,types,ratio_evals=5,max_evals=-1,nfolds=3,tolerance=-1,n_jobs=1,max_evals_optim=10,randomize=True,metric=corrNumba,population=100):

    points,results=gridSearch(model=model,param_grid=params,train=train,target=target,max_evals=max_evals,ratio_evals=ratio_evals,nfolds=nfolds,n_jobs=n_jobs,types=types,randomize=randomize,metric=metric)    
    final_params=hyperOptim(model=model,points=points,results=results,train=train,target=target,params=params,max_evals=max_evals_optim,lb=lb,ub=ub,tolerance=tolerance,nfolds=nfolds,types=types,metric=metric,population=population)    
    model.set_params(**final_params)    
    
    return calcMetricsRegression(model,train,target,nfolds),model

def makeConstraints(param_names,constraints):
    lb=[]
    ub=[]
    for k in param_names:
        lb.append(constraints[k][0])
        ub.append(constraints[k][1])


    return lb,ub

def assertConstraints(lb,ub,values):
    """
    gets the constraints back within the right range, if they are out of the range.
    :param lb:
    :param ub:
    :param values:
    :return:
    """
    dummy=[]
    for v,l,u in zip(values,lb,ub):
        if v<l:
            v=l
        elif v>u:
            v=u
        
        if type(l)==bool or type(u)==bool:
            v=bool(np.round(v))
        dummy.append(v)
    return dummy


def hyperOptim(model,points,results,train,target,param_grid,constraints,types,
               n_folds=5,max_evals=20,tolerance=-1,n_jobs=1,metric=corrNumba,
               population=100,optimizer=rfCMAOptim_helper):
    """
    Runs a hyperparameter optimization routine. The model is executed a number of N times, 
    the hyperparameter surface is learned by an ML algorithm (default random forest)
    and then the model is optimized by using a method (defauly CMA) 
    to find an optimal set of parameters, based on the model of the learned 
    hyperparameters. The proposed
    parameters are tried, they are added as a datapoint and the process is repeated.

    based on ideas first exposed at: https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
    """
#    try:
#        points=points.tolist()
#    except:
#        pass

#    try:
#        results=results.tolist()
#    except:
#        pass

    param_names=points.columns.tolist()
    lb,ub=makeConstraints(param_names=param_names, constraints=constraints)
    for i in range(0,max_evals):
        print("******************************************************")
        print('runnin hyperoptim evaluation number '+str(i) +' out of a total of '+str(max_evals))
        params_optimized=optimizer(points,results,max_evals,lb,ub,population)
        values=params_optimized[0]
        values=assertConstraints(lb,ub,values)
        print('proposal has a predicted score of:'+str(params_optimized[1]))
        print('proposal is : '+str(param_names)+" : "+str(values))
        new_points,new_results=runModel(model=model,params=pd.DataFrame([values],columns=param_names),train=train,target=target,n_folds=n_folds,n_jobs=n_jobs,types=types,metric=metric)
        print("new points are: " + str(new_points))
        points=points.append(new_points, ignore_index=True)
        results.append(new_results)

        delta=abs(new_results-max(results))
        if delta<tolerance:
            print("delta below tolerance. stopping optimization.")
            break

        print('best result is :'+str(max(results)))

    points=np.array(points)
    results=np.array(results)
    best_values=points[np.where(results==results.max())[0][0]]
    settings=OrderedDict(zip(param_names,best_values))

    if len(types)>0:
        for t,fun in types.items():
            settings[t]=fun(settings[t])

    return settings