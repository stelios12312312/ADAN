# -*- coding: utf-8 -*-
import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
from deap.algorithms import varAnd
from adan.aiem.genetics.evaluators import *
import array
import random as traditional_random

#import pathos

import pathos
import operator

#from adan import functions
from adan.functions import *
from adan.aidc.feature_selection import *

import time


def eaSimple_island(population,toolbox, cxpb, mutpb, ngen,halloffame=None, 
                    verbose=__debug__,allowed_time=np.inf, stats=None,FREQ=None,
                    percentage_migration=0.1):
    
    """
    ngen is used both for the total generations and for the within island generatins.
    So, the total number of gens will be ngen**2.
    
    FREQ: How often migration takes place. If FREQ=None, then it is set to ngen/3
    """
    
    #FREQ is how often migration takes place
    if FREQ is None:
        FREQ=int(ngen/3)
        if FREQ<0:
            FREQ=1
    
    toolbox.register("algorithm", eaSimple_timed, toolbox=toolbox, 
                     cxpb=cxpb, mutpb=mutpb, ngen=ngen, 
                     verbose=verbose,stats=stats,halloffame=halloffame)
    
    islands = population
    #The GA runs each time for ngen, and then it runs for a total number of equal to ngen/FREQ

    for i in range(0, ngen):
        start = time.time()
        results = toolbox.map(toolbox.algorithm, islands)
        islands = [pop for pop, logbook in results]
        
        if i % FREQ ==0:    
            print('******MIGRATION TAKING PLACE******')
            tools.migRing(islands, int(percentage_migration*len(islands[0])), tools.selBest)
        
        end = time.time()
        
        if (end-start)>allowed_time:
            if verbose:
                print('Time-out. Maximum allowed time exceeded.')
            break
    
    return islands

def eaSimple_timed(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__,allowed_time=np.inf):
    """This is a copy of the eaSimple() method from DEAP, but adjusted 
    to support time-out. In case of timeout, the most recent generation is 
    returned.
    
        
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        #-2 is the 'fail' value (e.g. the fitness function couldn't be computed)
        if fit is None:
            fit=(-2,)
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
        
    start = time.time()
    # Begin the generational process
    for gen in range(1, ngen+1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))
        
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            if fit is None:
                fit=(-2,)
            ind.fitness.values = fit
        
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)
            
        # Replace the current population by the offspring
        population[:] = offspring
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream) 
            
        end = time.time()
        
        if (end-start)>allowed_time:
            if verbose:
                print('Time-out. Maximum allowed time exceeded.')
            break
    
    return population, logbook
    

def calcNewFeatures(result_set,df,features='best'):
    """
    returns the best features alongside the variables participating in the complex variables
    """
    all_features=[]
    complex_features=[]
    pset=setPset(df)
    toolbox = base.Toolbox()
    toolbox.register("compile", gp.compile, pset=pset) 
    
    complex_columns=[]  
    all_columns=[]
    simple_columns=[]
    
    if features=='best':
        dummy='best_individuals_object'
    elif features=='all':
        dummy='all_features_individuals_object'
    
    for feat in result_set[dummy]:
        complex_features.append(toolbox.compile(feat))
        all_features.append(toolbox.compile(feat))
        complex_columns.append(str(feat))
        all_columns.append(str(feat))
    
    simple_features=[]
    
    for feat in result_set['variables']:
        simple_features.append(df[feat])
        simple_columns.append(str(feat))
        all_features.append(df[feat])
        all_columns.append(str(feat))
        
    
    return pd.DataFrame(np.column_stack(all_features),columns=all_columns),pd.DataFrame(np.column_stack(complex_features),columns=complex_columns),pd.DataFrame(np.column_stack(simple_features),columns=simple_columns)
        
    

def setPset(df):
    pset = gp.PrimitiveSet("MAIN", 0,prefix="coef")
    pset.addPrimitive(add,2)
    pset.addPrimitive(sub, 2)
    pset.addPrimitive(mul, 2)
    pset.addPrimitive(div, 2)
    
    for fun in singlefunctions:
        pset.addPrimitive(fun,1)
    
    for col in df.columns.values:
        #we must use strings for column names otherwise the functions interpret the
    #column names as numbers
        pset.addTerminal(df[col].values,name=col)
        
    return pset



def findFeaturesGP(df,target,population=300,ngen=50,cxpb=0.9,features=-1,
                   max_tree=3,evaluator=evalPearsonCorNumba,
                   task="regression",n_processes=1,allowed_time=None,target_sampling=0.8):
                       
    """
    This function calculates complex features that correlate with the response variable.
    Output:
    
    A dictionary with the following fields:
    
    best_features: a list of lists, where every element is a feature selected by the best n features as defined by the cbf method
    best_features_plus_cols: a list of lists, where every element is a feature selected by the best n features as defined by the cbf method plus
    any original features participating in the creation of the individuals 
    best_individuals_equations: the equations used to compute the best_features (this is the string version of best_individuals_object)
    best_individuals_plus_columns: like the previous, plus the column names of the individual features
    best_individuals_object: the programs used to compute the best_features
    
    scores: the score of each individual produced during the genetic programming
    scores_cbf: the cbf score of each feature (all features not just the best ones)
    variables: the names of the original variables that participate in the creation of the features in the best_features
    all_features: a list of lists with all the features produced by the genetic algorithm
    all_features_individuals: the programs used to compute all_features
    
    features: if features<1, then the algorithm simply defaults to 1
    target_sampling: When the features are evaluated, we can sample a % of the targets, and evaluate
    the performace on this subset. This should help with overfitting and finding better solutions.
    
    """
    
    if features<1:
        features=1
    
    if task=='regression' and evaluator==None:
        evaluator=evalPearsonCorNumba
    elif task=='classification' and evaluator==None:
        evaluator=evalANOVANumba
    
    mutpb=1-cxpb    
    
    # for col in df.columns:
    #     df[col]=df[col].astype('float64')
    
    pset=setPset(df)
            
    creator.create("FitnessMax", base.Fitness, weights=(1,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=max_tree)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset) 
    
    #need to do that because multithreading does not support functions with more than one arguments
    def evaluate(x):
        return evaluator(x,toolbox=toolbox,targets=target,sampling=target_sampling)
    
    #toolbox.register("evaluate", evaluator,toolbox=toolbox, targets=targets)
    toolbox.register("evaluate", evaluate)

    #toolbox.register("select", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selNSGA2)
    toolbox.register("select", tools.selDoubleTournament,fitness_size=3,parsimony_size=1.4,fitness_first=True)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=max_tree)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=max_tree)) 
    
    if type(population)==type([]):  
        toolbox.register("deme", tools.initRepeat, list, toolbox.individual)
        DEME_SIZES = population
        pop = [toolbox.deme(n=i) for i in DEME_SIZES]
        hof = tools.HallOfFame(sum(population))
    else:
    
        pop = toolbox.population(n=population)
        hof = tools.HallOfFame(population)
    
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    

    
    
    if n_processes>1:
        pool = pathos.multiprocessing.ProcessingPool(n_processes)
        toolbox.register("map", pool.map)
    
#    pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu,lamb, cxpb,mutpb,ngen=ngen, stats=mstats,
#                                   halloffame=hof, verbose=True)
#    if allowed_time is None:
#        pop, log = algorithms.eaSimple(pop, toolbox, cxpb,mutpb, ngen=ngen, stats=mstats,
#                                       halloffame=hof, verbose=True)
    if type(population)==type([]):
        pop = eaSimple_island(pop, toolbox, cxpb,mutpb, ngen=ngen, stats=mstats,
                                       halloffame=hof, verbose=True, allowed_time=allowed_time)
    else:
        pop, log = eaSimple_timed(pop, toolbox, cxpb,mutpb, ngen=ngen, stats=mstats,
                                       halloffame=hof, verbose=True, allowed_time=allowed_time)
                         
    allfeatures=[]
    allfeatures_individuals_object=[]
    scores=[]
    feature_names=[]
    best_individuals_object=[]
    for i in range(0,len(hof.items)):
        # print(hof.items[i])
        feature=toolbox.compile(hof.items[i])     
        
        if not np.isnan(feature).any():
            #need to guard against zero variance features
            if np.var(feature)>0.0:
                allfeatures.append(feature)
                allfeatures_individuals_object.append(hof.items[i])
                feature_names.append(str(hof.items[i]))
                best_individuals_object.append(hof.items[i])
                #for some reason in DEAP the key in the hall-of-fame is the score
    
#    if features>0:
#        cbfscores=cbfSelectionNumba(allfeatures,target,task=task)
#        bestindices=sorted(range(len(cbfscores)), key=lambda x: cbfscores[x],reverse=True)
#    else:
#        cbfscores=np.ones(len(allfeatures))
#        bestindices=range(len(allfeatures))
                
    cbfscores=cbfSelectionNumba(allfeatures,target,task=task)
    bestindices=sorted(range(len(cbfscores)), key=lambda x: cbfscores[x],reverse=True)

    bestfeatures=[]
    bestindividuals=[]
    bestindividuals_plus_cols=[]
    scorescbf=[]
    best_features_plus_cols=[]
    best_individuals_object_final=[]
    for i  in range(0,int(features)):
        index=bestindices[i]
        bestfeatures.append(allfeatures[index])
        best_features_plus_cols.append(allfeatures[index])
        
        bestindividuals.append(feature_names[index])
        bestindividuals_plus_cols.append(feature_names[index])
        best_individuals_object_final.append(best_individuals_object[i])
        
        # scores.append(eval(str(hof.keys[index])))
        # scorescbf.append(cbfscores[index])
               
       
    #all features includes the best variables, plus any single variables which might participate in the creation of the complex variables
    final_vars=[] 
    str_individuals=str(bestindividuals)
    for col in df.columns:
        if str_individuals.find(col)>-1:
            final_vars.append(col)
            #append the original variable to bestfeatures if it exists in a complex feature
            best_features_plus_cols.append(df[col].values)
            bestindividuals_plus_cols.append(col)
                           
    #combine all features (individual and composite) into one df       
    best_all_feats_df=pd.DataFrame(np.column_stack(best_features_plus_cols),columns=bestindividuals_plus_cols)
    
    return {'best_features':bestfeatures,'best_features_plus_cols':best_features_plus_cols,
            'best_individuals_equations':bestindividuals,'best_individuals_object':best_individuals_object_final,
    'scores':scores,'scores_cbf':scorescbf,'variables':final_vars,
    'all_features':allfeatures,'all_features_individuals_object':allfeatures_individuals_object,'best_all_feats_df':best_all_feats_df}


def findEquationFeatures(features_to_be_used,task,target,ngen=10,population=10,crossover_prob=0.5,mut_prob=0.1,individual_mut=0.1,tournsize=3):
    """
    Performs feature selection over the set of features before doing the
    symbolic modelling
    
    individual_mut: If a mutation occurs, then each item might be flipped according to this probability
    """
    
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    import array
    creator.create("Individual", array.array, typecode='b', fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Attribute generator
    toolbox.register("attr_bool", traditional_random.getrandbits,1)
    
    # Structure initializers
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, features_to_be_used.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    #we import here to avoid a cyclical import
    from adan.aiem.symbolic_modelling import findSymbolicExpressionL1_regression_helper, findSymbolicExpressionL1_classification_helper

    def evalOneMax(individual):
        if sum(individual)==0:
            return -100,
        else:
            ind=np.array(individual,bool)
        if task=='regression':
            models=findSymbolicExpressionL1_regression_helper(features_to_be_used.loc[:,ind].values,target)
        elif task=='classification':
            models=findSymbolicExpressionL1_classification_helper(features_to_be_used.loc[:,ind].values,target)  
        
        performances=[perf[1] for perf in models]
        maximum=max(performances)
        
        return maximum,
    
    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=individual_mut)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)

    
    pop = toolbox.population(n=population)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, 
                                   mutpb=mut_prob, ngen=ngen, 
                                   stats=stats, halloffame=hof, verbose=True)
    final_choice=hof.items[0]
    final_choice=np.array(final_choice,bool)
    
    #return pop, log, hof
    return final_choice