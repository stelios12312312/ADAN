#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 18:18:11 2019

@author: stelios
"""

import array
import random
import json

import numpy

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import numpy as np
import pandas as pd

# gr*.json contains the distance map in list of list style in JSON format
# Optimal solutions are : gr17 = 2085, gr24 = 1272, gr120 = 6942

def generate_datapoint(df):
    point=df.sample(1).iloc[0,:].tolist()
    
    return point

def evaluate_model(model,original_dataframe,datapoint,reference_class=None):
    
    datapoint=pd.DataFrame(datapoint).T
    datapoint.columns=original_dataframe.columns
    result=model.evaluate(datapoint)
    
    if reference_class is not None:
        result=(result[0][reference_class],)
    
    return result


def mutate(individual,original_dataframe,mutprob):
    datapoint=individual
    #datapoint=pd.DataFrame(individual).T
    #datapoint.columns=original_dataframe.columns 
    
    for i in range(len(datapoint)):
        if random.random()<mutprob:
            if random.random()<0.5 or type(datapoint[i])==type('str'):
                sampled=original_dataframe.sample(1)
                datapoint[i]=sampled.iloc[0,i]
            else:
                datapoint[i]=datapoint[i]+(np.random.randn()+original_dataframe.iloc[:,i].mean())*original_dataframe.iloc[:,i].std()

    
    return datapoint,


def feasible(individual,original_dataframe,constraints={}):
    """Feasibility function for the individual. Returns True if feasible False
    otherwise."""
    df=original_dataframe
    maxes=original_dataframe.max().tolist()
    mins=original_dataframe.min().tolist()
    
    for i in range(len(individual)):
        if type(individual[i])!='str':
            if individual[i]>maxes[i] or individual[i]<mins[i]:
                return False
     
    #Handle constraints separately
    for const in constraints.keys():
        column_index=np.where(df.columns==const)[0][0]
        values=constraints[const]
        
        current_value=individual[column_index]
        
        if type(values[0])==type('str'):
            if current_value not in values:
                return False
        else:
            if current_value<values[0]:
                return False
            if current_value>values[1]:
                return False
           
    return True
    

def distance(individual,original_dataframe,constraints={},category_penalty=10):
    """A distance function to the feasibility region.
    
    category_penalty: In case the variable is categorical, then this penalty is applied. Otherwise
    we simply apply the quadratic difference.
    
    """
    
    df=original_dataframe
    maxes=original_dataframe.max().tolist()
    mins=original_dataframe.min().tolist()
    
    penalty=0
    
    for i in range(len(individual)):
        if type(individual[i])!='str':
            if individual[i]>maxes[i]: 
                penalty+=(maxes[i] - individual[i])**2
            if individual[i]<mins[i]:
                penalty+=(mins[i] - individual[i])**2
    
    #Handle constraints separately
    for const in constraints.keys():
        column_index=np.where(df.columns==const)[0][0]
        values=constraints[const]
        
        current_value=individual[column_index]
        
        if type(values[0])==type('str'):
            if current_value not in values:
                penalty+=category_penalty
        else:
            if current_value<values[0]:
                penalty+=(penalty-values[0])**2
            if current_value>values[1]:
                penalty+=(penalty-values[1])**2
                
    return penalty
    
    

def reverse_optimise(original_dataframe,model,target_name,constraints={},fitness='min',
                     reference_class=None,ngen=10,crossover=0.7,mutation=0.2):
    """
    original_dataframe: The original dataframe that was used to build the model
    
    model: The trained model
    
    target_name: The name of the target variable
    
    constraints: A dictionary of constraints. The constraints for numerical variables, should be a tuple
    of the form (min,max). For categorical variables it should be a list of valid categories.
    
    fitness: acceptable values are min or max
    
    reference_class: In case of a classification problem you need to provide 
        a reference class, as an index, e.g. 0, 1, 2, etc.
    
    """
    
    df=original_dataframe
    target=df[target_name]
    df.drop(target_name,axis=1,inplace=True)
    
    
    if fitness=='min':
        creator.create("FitnessObjective", base.Fitness, weights=(-1.0,))
    elif fitness=='max':
        creator.create("FitnessObjective", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessObjective)
    
    toolbox = base.Toolbox()
    
    # Generate the initial datapoints for an individual
    toolbox.register("single_datapoint", generate_datapoint)
    
    # Structure initializers
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda : toolbox.single_datapoint(df))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)


    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", lambda individual: mutate(individual,df,0.1))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda datapoint: evaluate_model(model,df,datapoint,reference_class=reference_class))
    
    #We need to adjust the delta in the penalty function, depending on whether we minimise or maximise.
    if target.dtype=='object':
        if fitness=='min':
            delta=1
        elif fitness=='max':
            delta=0
    else:
        if fitness=='min':
            delta=max(target)
        elif fitness=='max':
            delta=min(target)
    
    #We need to adjust the delta in the penalty function, depending on whether we minimise or maximise.
    if fitness=='min':
        toolbox.decorate("evaluate", tools.DeltaPenalty(lambda individual: feasible(individual,original_dataframe,constraints=constraints), 
                                                        delta, 
                                                        lambda individual: distance(individual,original_dataframe,constraints=constraints)))
    elif fitness=='max':
                toolbox.decorate("evaluate", tools.DeltaPenalty(lambda individual: feasible(individual,original_dataframe,constraints=constraints), 
                                                        delta, 
                                                        lambda individual: distance(individual,original_dataframe,constraints=constraints)))

    pop = toolbox.population(n=30)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    algorithms.eaSimple(pop, toolbox, crossover, mutation, ngen, stats=stats, 
                        halloffame=hof)
    
    #return pop, stats, hof
    solution=list(hof[0])
    datapoint=pd.DataFrame(solution).T
    datapoint.columns=original_dataframe.columns
    
    return datapoint

#import joblib
#model=joblib.load('model_regression_abalone.pkl')
#df=pd.read_csv('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/abalone/abalone.csv',header=None)
##test numeric constraints
#res=reverse_optimise(df,model,target_name=8,reference_class=None,fitness='max',
#                     ngen=5,constraints={1:(0.5,0.6),2:(0.5,0.6),0:('M')})

#model=joblib.load('abalone_model_classification.pkl')
#df=pd.read_csv('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/abalone/abalone.csv',header=None)
#res=reverse_optimise(df,model,target_name=0,reference_class=1,fitness='max',
#                     ngen=10,constraints={1:(0.5,0.6),2:(0.5,0.6)})