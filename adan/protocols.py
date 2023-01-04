#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 16:12:49 2019

@author: stelios
"""

# import adan
import sklearn
import numba
from adan.aiem.genetics import *
from adan.aidc.utilities import *
from adan.aidc.feature_selection import *

from adan.metrics.metrics_utilities import *


from adan.aiem.symbolic_modelling import *
from adan.aidc.utilities import *

from adan.aiem.symbolic_modelling import *
from adan.aidc.utilities import *
from adan.aiem.genetics.genetic_programming import *
from adan.aiem.symbolic_modelling import *

from adan.aist.mappers import *
from adan.aipd.aipd_main import *
from adan.aipd.causality import run_causality,interpret_causal_results
import sklearn
import copy


#pd.set_option('display.max_columns', 500)
# import random
#random.seed(27)
# pd.set_option('display.max_columns', 500)

#file that contains smart protocols (recipes) for quickly analyzing data
import numpy as np
from adan.aiem.genetics.genetic_programming import findFeaturesGP
from adan.aiem.symbolic_modelling import findSymbolicExpression, findAcceptable, Model
from adan.aidc.utilities import readData, prepareTrainData
from adan.aidc.feature_selection import chooseBest
import time
from functools import reduce


from lightgbm import LGBMClassifier,LGBMRegressor


def _find_constraints(df):
    """
    Returns the valid values for the input to a model, based on the dataframe, 
    as well as default settings.
    
    For numerical variables the valid range is the minimum and the maximum, and the 
    default range is the median.
    
    For categorical variables the valid variables are the pre-existing categories
    and the default value is the mode.
    
    valid categories are returned 
    """
    
    numerical_max=df.max()[df.dtypes!='object']
    numerical_min=df.max()[df.dtypes!='object']
    numerical_median=df.median()[df.dtypes!='object']
    categorical_mode=df.max()[df.dtypes=='object']
    
    categories_only=df.loc[:,df.dtypes=='object']
    
    #for some reason sometimes this fails
    # cats=categories_only.apply(np.unique,axis=0)
    
    cats={}
    if categories_only.shape[1]>0:
        categories_only=categories_only.astype('category')
        for category in categories_only.columns:
            cats[category]=categories_only[category].cat.categories
    
    
    return {'numerical_max':numerical_max,'numerical_min':numerical_min,'numerical_median':numerical_median,
            'categorical_mode':categorical_mode,'categories':cats}
    
    

def _symbolic_helper_params_generator(df):            
    max_tree=np.random.randint(2,5)
    population=np.random.randint(100,8*df.shape[1]**3)
    if population>10000:
        population=10000
    #features=np.round(population/2.0)
    features = np.random.randint(10,int(np.round(population/2.0)))
    if features>100:
        features=100
    ngen=int(np.random.randint(1,5)*0.1*population)

    return {'max_tree':max_tree,'population': population,'features': features,'ngen': ngen}

# def symbolic_regression_protocol(df,target,task,time_allowed=15*60,iterations=10,sep=',',
#                                  params=None,median_deviation_complexity=4,genetic_selection=True):
#     """
#     params: the parameters for the genetic algorithm
#     genetic_selection: If true, then there is one more round of genetic algorithms
#     after the features have been generated. This time the goal is to choose
#     the best subset of features that maximises performance.
#     """
    
#     df_initial=readData(df,sep=sep)
#     df, target,scaler,centerer,categorical_vars,filler,log,
#     target_category_mapper,numerical_cols,n_components,pca_object,bad_target_values=prepareTrainData(dataframe=df_initial,
#                                                               target_name=target,
#                                                               task=task,
#                                                               fix_skew=fix_skew)
    
#     all_results = []
#     total_time = float(time_allowed)

#     performances = []

#     for i in range(0,iterations):
#         print('\n\nprotocol iteration: '+str(i))
        
#         if total_time<=0:
#             print('total time exceeded')
#             break
        
#         if params==None:
#             params = _symbolic_helper_params_generator(df)
#         df2 = chooseBest(df,target,limit_n=1000,method="num",quant=(1-np.random.randint(1,9)*0.1))[0]
#         params['df']= df2
#         params['target']=target
#         params['allowed_time']=np.round(float(total_time)/iterations)
        
#         start = time.time()
        
#         g = findFeaturesGP(**params)
#         print('starting symbolic expression search')
#         if not genetic_selection:
#             res = findSymbolicExpression(df2,target,g,scaler,task=task)
#         else:
#             res = findSymbolicExpression_genetic(df2,target,g,scaler,task=task)

#         all_results = all_results + res
        
#         end = time.time()
#         time_diff=end-start
#         total_time = total_time - float(time_diff)
        
            
#     clean_results = findAcceptable(all_results,median_deviations=median_deviation_complexity)
    
#     final_models=[]
#     for m in clean_results:
#         final_models.append(Model(model=m[0],performance=m[1],genetic_result_set=g,
#                                   scaler=scaler,
#                                   categorical_vars=categorical_vars,filler=filler,
#                                   pca_object=pca_object,numerical_variables=numerical_cols))
    
#     #return clean_results, all_results, df2, g
#     return final_models,clean_results

def tidyup_find_best(find_symbolic_expression_result,features_GP_result,scaler,
                     centerer,categorical_vars,filler,task,target,
                     target_category_mapper,numerical_variables,components,pca_object):
    """
    find_symbolic_expression_result: the result of findSymbolicExpression()
    features_GP_result: the result of a GP search for features
    returns:
        the best model and a list of model objects
        
    performance is r2 score for regression and cohen's kappa for classification
    """
    perfs=[]
    for c in find_symbolic_expression_result:
        print("performance of model is:"+str(c[1]))
        perfs.append(c[1])
    perfs=np.array(perfs)
    
    final_models=[]
    for m in find_symbolic_expression_result:
        original_results=m[2]
        final_models.append(Model(m[0],m[1],features_GP_result,scaler,centerer,categorical_vars,filler,task,
                                  target=target,sklearn_model=m[3],original_results=m[2],
                                  target_category_mapper=target_category_mapper,
                                  numerical_variables=numerical_variables,pca_object=pca_object,
                                  components=components,
                                  eq_breakdown=m[4]))
    
    model = final_models[np.where(perfs==max(perfs))[0][0]]
    return model,final_models    


def _test_quality_results(results,model,task,target):
    """
    In some cases, due to the symbolic regression containing terms like Exp, etc.
    the results might be NaN. In those cases, ADAN is return the mean (regression)
    or the mode (classification) of the target variable. If this happens too many times
    then it is wise to re-run the algorithm. This function has some hard-coded
    tolerance levels,and will return False is the error exceeds acceptable limits.
    """
    successful_execution=True
    
    original_res=model._original_results
    #original_res=target

    if task=='classification':
        #if the problem is multiclass
        if len(results.shape)>1:
            if sum(np.argmax(results,axis=1)==original_res)/len(original_res)<0.95:
                successful_execution=False
                print('ISSUES WHEN COMPARING SKLEARN RESULTS AND ADAN RESULTS')
                return False
            if results.shape[1]!=len(np.unique(target)):
                sentence='The model predicts less classes than originally exist. Consider re-running.'
                results_log+=sentence
                print(sentence)
        else:
            #results=np.ndarray.flatten(results.values)
            if sum(np.round(results)==original_res)/len(original_res)<0.95:
                successful_execution=False
                print('ISSUES WHEN COMPARING SKLEARN RESULTS AND ADAN RESULTS')
                return False

    else:
        #for the housing dataset, the original results are less than the dataframe.
        #the reason is that the target in the original contains some NAs
        try:
            #we accept only an error of 0.1% over the mean of the original target variable
            mean_predictor=np.mean(abs(original_res-np.mean(original_res)))
            abs_error=np.mean(abs(results-original_res))
            if abs_error>mean_predictor:
                successful_execution=False
                print('ISSUES (POTENTIALLY NUMERICAL) WHEN COMPARING SKLEARN RESULTS AND ADAN RESULTS')
                print('absolute error vs mean predictor error: {0} vs {1}'.format(abs_error,mean_predictor))
                return False
        except:
            pass
        
    return successful_execution
    
    

def run_equation_model(df,target_name,task,ngen,max_tree,
             n_pop,n_features,allowed_time,test_perc=0.15,n_processes=1,
             complexity_tolerance=1.5,choice_method='quant',quant_or_num=0.5,fix_skew=True,
             limit_n_for_feature_selection=100,target_sampling=0.8,
             pca_criterion_thres=0.05,test_indices=None,genetic_selection=True,
             ngen_for_second_round=10,npop_for_second_round=10,crossover_second_round=0.5,
             mut_prob_second_round=0.1,individual_mut_second_round=0.1,selection_variable_ratio=0.1,
             fix_margins_low=False,fix_margins_high=False):
    
    """
    df: A pandas dataframe
    
    target_name: The name of the target variable
    
    ngen: The number of generations for the genetic algorithm
    
    max_tree: The maximum depth of the tree used by the genetic programming algorithm
    
    n_pop: The population. This taks a list because we are using the islands version of genetic algorithms
    
    n_features: The final number of features to ue for the equation
    
    store_results: Where to save the results
    
    allowed_time: The allowed execution time of the genetic algorithm in minutes. The actual
    running time might be slightly longer due to some additional processing that need to be done.
    
    test_perc: ADAN can perform a train/test split. The determines the % for the test split.
    
    choice_method: shall the feature chooser choose features based on a maximum number of features
    or their score as a quantile?
    complexity_tolerance: create a threshold= std of complexities times this value.
    If complexity penalty is lower than this threshold, then discard the solution
    
    allowed_time: Time allowed to spend on the genetic programming feature creation (in seconds)
    
    The 'second_round' variables refer to the second round of optimisation which takes place
    if genetic_selection=True. In this case, all the features are used, and then they are selected
    through a genetic algorithm. So, the algorithm is trying to find the best subset of features
    that gives the highest performance.
    
    target_sampling: When the features are evaluated, we can sample a % of the targets, and evaluate
    the performace on this subset. This should help with overfitting and finding better solutions.
    
    selection_variable_ratio: Only perform feature selection numcolumns/numrows is above this ratio
    
    selection_variable_threshold: This is defined as (number of variables)/(number of rows). If this
    ratio is above the threshold, then perform feature selection. Feature selection can be very expensive, 
    so, it is not worth doing when there are many more rows than columns.
        
    """
    

    if test_perc>0 or test_indices is not None:
         #Create the test data
        if test_indices is None:
            test_indices=np.random.choice(np.arange(0,df.shape[0]),int(test_perc*df.shape[0]),replace=False)
        df_test=df.iloc[test_indices,:]
        df=df.drop(index=test_indices,axis=0)       
        try:
        #this is only for classification, some categories might have trailing white space
            target_test=df_test[target_name].str.strip()
        except:
            target_test=df_test[target_name]
            
        #If there are many categories, there is the risk of some of them not showing up in the
        #training or test set. Hence, do the resampling again
        trial=0
        if task=='classification':
            try:
                 #this is only for classification, some categories might have trailing white space
                target_train_dummy=df_train[target_name].str.strip()
            except:
                target_train_dummy=df_test[target_name]
            

            #Create the test data
            test_indices=np.random.choice(np.arange(0,df.shape[0]),int(test_perc*df.shape[0]),replace=False)
            df_test=df.iloc[test_indices,:]
            df=df.reset_index(drop=True).drop(index=test_indices,axis=0)   
            
            try:
                #this is only for classification, some categories might have trailing white space
                target_test=df_test[target_name].str.strip()
            except:
                target_test=df_test[target_name]

            
    # for col in df.columns:
    #     df[col]=df[col].astype('float64')
    #this is where the cleaning takes place
    #this function prepares the data and returns the following: a processed dataframe,
    #processed target variable (removed y values if they are empty), centerer scaler and cat vars
    #are used for preprocessing, as well as the filler. The log describes changes that took place in the vars.
    df2, target,scaler,centerer,categorical_vars,filler,log,\
    target_category_mapper,numerical_cols,components,pca_object,bad_target_values=\
    prepareTrainData(dataframe=df,target_name=target_name,task=task,fix_skew=fix_skew,pca_criterion_thres=pca_criterion_thres)
                                                                                                 
    #feature selection
    ratio=df2.shape[1]/df2.shape[0]
    if (ratio)>selection_variable_ratio:
        print('original shape before feature selection: '+str(df2.shape))
        #done for compatibility purposes
        df2.dtype=df2.dtypes
        df2=chooseBest(df2,target,limit_n=limit_n_for_feature_selection,method=choice_method,quant=quant_or_num)[0]
    else:
        print('No feature selection will be performed. The variable to rows ratio is :'+str(ratio))
    
    print('shape after feature selection: '+str(df2.shape))
    
    #Create the genetic programming features
    g=findFeaturesGP(df=df2,target=target,ngen=ngen,max_tree=max_tree,population=n_pop,
                     features=n_features,n_processes=n_processes, 
                     allowed_time=allowed_time,target_sampling=target_sampling)
    
    
    if not genetic_selection:
        res_symbolic = findSymbolicExpression(df2,target,g,scaler,task=task)
    else:
        res_symbolic,final_features= findSymbolicExpression_genetic(df2,target,g,scaler,task=task,
                                                      ngen=ngen_for_second_round,
                                                      population=npop_for_second_round,
                                                      crossover_prob=crossover_second_round,
                                                      mut_prob=mut_prob_second_round,
                                                      individual_mut=individual_mut_second_round)
        

    #Clean up the results of the last function call findSymbolicExpression()
    model,final_models = tidyup_find_best(res_symbolic,g,scaler,centerer,categorical_vars,filler,
                                          task=task,target=target,
                                          target_category_mapper=target_category_mapper,
                                          numerical_variables=numerical_cols,
                                          pca_object=pca_object,components=components)
    
    ##################################
    #THIS CODE WAS USED WHEN DEBUGGING FOR A PARTICULAR PROBLEM WHERE
    #SKLEARN VALUES WERE DIFFERENT TO ADAN VALUES. THIS WAS FIXED
    #SOME SMALL DEVIATIONS ARE NORMAL DUE TO NUMERICAL ISSUES
    
    #evaluate differences between predicted values and the sklearn model as a
    #sanity check. There should be some differences due to numerical reasons, but 
    #these should be very small
   
    
    #Need to use only finite target values, otherwise sometimes there are deviations in the 
    #sizes of hte dataset
    
    # ska1=model.evaluate(df[np.logical_not(df[target_name].isnull())])
    # skld=model._sklearn_model.predict(final_features)
    
    # if task=='classification':
    #     try:
    #         #if multiclass then convert to a single array, otherwise pass
    #         ska1=np.argmax(ska1,axis=1)
    #     except:
    #         pass
    #in some cases classification differences can be large, because we are dealing
    #with binary outcomes, so the max difference can be 1 for example
    # dif1=pd.Series(abs(ska1-skld))
    # print('difference between model and sklearn')
    # print(dif1.max())

    ##############################

    
    realizer=sentenceRealizerSymbolic()
    realizer.interpretSymbolic(res_symbolic,task=task)
    res_sentence=realizer.realizeAll()
    
    #This is the code for testing on the test data
    if test_perc>0:
        results_test=model.evaluate(df_test,fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
        if task=='classification':
            for mapping in target_category_mapper.keys():
                target_test.values[target_test==mapping]=target_category_mapper[mapping]
            try:
                #multiclass
                performance_test=sklearn.metrics.cohen_kappa_score(target_test.tolist(),np.argmax(results_test,axis=1))
            except:
                performance_test=sklearn.metrics.cohen_kappa_score(target_test.tolist(),np.round(results_test).astype(np.int32))
        else:
            #sometimes there are NaNs in the target
            there_are_nans=np.where(np.isnan(target_test))[0]
            df_test=df_test.reset_index().drop(there_are_nans)
            target_test=np.delete(target_test.values,there_are_nans)
            results_test=np.delete(results_test.values,there_are_nans)
            performance_test=sklearn.metrics.r2_score(target_test,results_test)
    else:
        performance_test=None
    
    if bad_target_values is not None:
        train_results=model.evaluate(df.reset_index(drop=True).drop(bad_target_values),fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
    else:
        train_results=model.evaluate(df,fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
    
    if test_perc>0:
        test_results=model.evaluate(df_test,fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
    else:
        test_results=None
        target_test=None
        
    #Sometimes the model screws up and we have to circle around the other models
    successful_execution=_test_quality_results(train_results,model,task=task,
                                               target=target)    
    
    
    i=0
    while not successful_execution and i<=len(final_models):
        print('Initial model unsuccessful. Rotating other final models...')
        model2=final_models[i]

        if bad_target_values is not None:
            train_results=model2.evaluate(df.reset_index(drop=False).drop(bad_target_values),fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
        else:
            train_results=model2.evaluate(df,fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
            
        successful_execution=_test_quality_results(train_results,model2,task=task,
                                                   target=target)
        i+=1
        if i==len(final_models):
            print('ISSUES RUNNING MODELS. NO VALID MODEL FOUND.')
            break
        
        
    #Now that we chose a model, let's estimate the training performance
    if task=='classification':
        for mapping in target_category_mapper.keys():
            target_test.values[target_test==mapping]=target_category_mapper[mapping]
        try:
            #multiclass
            performance_train=sklearn.metrics.cohen_kappa_score(target.tolist(),np.argmax(train_results,axis=1))
        except:
            performance_train=sklearn.metrics.cohen_kappa_score(target.tolist(),np.round(train_results).astype(np.int32))
    else:
        #sometimes there are NaNs in the target
        performance_train=sklearn.metrics.r2_score(target,train_results)    

    results={'model':model,'final_models':final_models,'issues_log':log,
             'natural_language_interpretation':res_sentence,'ground_truth_train':target,
             'ground_truth_test':target_test,
             'performance_test':performance_test,'successful_execution':successful_execution,
            'performance_train':performance_train,'predicted_train_values':train_results,
            'predicted_test_values':test_results,
            'components':components,'processed_input_data':df2,'sentence_realizer':realizer,
            'dataset_score':realizer.results['dataset_score']}
    
    return results

def run_equation_model_cv(df,target_name,task,ngen,max_tree,
             n_pop,n_features,allowed_time,test_perc=0.15,n_processes=1,
             complexity_tolerance=1.5,choice_method='quant',quant_or_num=0.5,fix_skew=True,
             limit_n_for_feature_selection=100,target_sampling=0.8,
             pca_criterion_thres=0.05,test_indices=None,genetic_selection=True,
             ngen_for_second_round=10,npop_for_second_round=10,crossover_second_round=0.5,
             mut_prob_second_round=0.1,individual_mut_second_round=0.1,selection_variable_ratio=0.1,
             fix_margins_low=False,fix_margins_high=False,n_folds=10):
    
    """
    CROSS-VALIDATION VERSION
    
    df: A pandas dataframe
    
    target_name: The name of the target variable
    
    ngen: The number of generations for the genetic algorithm
    
    max_tree: The maximum depth of the tree used by the genetic programming algorithm
    
    n_pop: The population. This taks a list because we are using the islands version of genetic algorithms
    
    n_features: The final number of features to ue for the equation
    
    store_results: Where to save the results
    
    allowed_time: The allowed execution time of the genetic algorithm in minutes. The actual
    running time might be slightly longer due to some additional processing that need to be done.
    
    test_perc: ADAN can perform a train/test split. The determines the % for the test split.
    
    choice_method: shall the feature chooser choose features based on a maximum number of features
    or their score as a quantile?
    complexity_tolerance: create a threshold= std of complexities times this value.
    If complexity penalty is lower than this threshold, then discard the solution
    
    allowed_time: Time allowed to spend on the genetic programming feature creation (in seconds)
    
    The 'second_round' variables refer to the second round of optimisation which takes place
    if genetic_selection=True. In this case, all the features are used, and then they are selected
    through a genetic algorithm. So, the algorithm is trying to find the best subset of features
    that gives the highest performance.
    
    target_sampling: When the features are evaluated, we can sample a % of the targets, and evaluate
    the performace on this subset. This should help with overfitting and finding better solutions.
    
    selection_variable_threshold: This is defined as (number of variables)/(number of rows). If this
    ratio is above the threshold, then perform feature selection. Feature selection can be very expensive, 
    so, it is not worth doing when there are many more rows than columns.
    
    """
    
    df.index=np.arange(0,df.shape[0])

    # for col in df.columns:
    #     df[col]=df[col].astype('float64')
    #this is where the cleaning takes place
    #this function prepares the data and returns the following: a processed dataframe,
    #processed target variable (removed y values if they are empty), centerer scaler and cat vars
    #are used for preprocessing, as well as the filler. The log describes changes that took place in the vars.
    df2, target,scaler,centerer,categorical_vars,filler,log,\
    target_category_mapper,numerical_cols,components,pca_object,bad_target_values=\
    prepareTrainData(dataframe=df,target_name=target_name,task=task,fix_skew=fix_skew)
                                                                                                 
    #feature selection
    ratio=df2.shape[1]/df2.shape[0]
    if (ratio)>selection_variable_ratio:
        print('original shape before feature selection: '+str(df2.shape))
        df2=chooseBest(df2,target,limit_n=limit_n_for_feature_selection,method=choice_method,quant=quant_or_num)[0]
    else:
        print('No feature selection will be performed. The variable to rows ratio is :'+str(ratio))
    
    print('shape after feature selection: '+str(df2.shape))
    
    #Create the genetic programmig features
    g=findFeaturesGP(df=df2,target=target,ngen=ngen,max_tree=max_tree,population=n_pop,
                     features=n_features,n_processes=n_processes, 
                     allowed_time=allowed_time,target_sampling=target_sampling)
    
    
    if not genetic_selection:
        res_symbolic = findSymbolicExpression(df2,target,g,scaler,task=task)
    else:
        res_symbolic,final_features= findSymbolicExpression_genetic(df2,target,g,scaler,task=task,
                                                      ngen=ngen_for_second_round,
                                                      population=npop_for_second_round,
                                                      crossover_prob=crossover_second_round,
                                                      mut_prob=mut_prob_second_round,
                                                      individual_mut=individual_mut_second_round)
        

    #Clean up the results of the last function call findSymbolicExpression()
    model,final_models = tidyup_find_best(res_symbolic,g,scaler,centerer,categorical_vars,filler,
                                          task=task,target=target,
                                          target_category_mapper=target_category_mapper,
                                          numerical_variables=numerical_cols,
                                          pca_object=pca_object,components=components)
    
    ##################################
    #THIS CODE WAS USED WHEN DEBUGGING FOR A PARTICULAR PROBLEM WHERE
    #SKLEARN VALUES WERE DIFFERENT TO ADAN VALUES. THIS WAS FIXED
    #SOME SMALL DEVIATIONS ARE NORMAL DUE TO NUMERICAL ISSUES
    
    #evaluate differences between predicted values and the sklearn model as a
    #sanity check. There should be some differences due to numerical reasons, but 
    #these should be very small
   
    
    #Need to use only finite target values, otherwise sometimes there are deviations in the 
    #sizes of hte dataset
    
    # ska1=model.evaluate(df[np.logical_not(df[target_name].isnull())])
    # skld=model._sklearn_model.predict(final_features)
    
    # if task=='classification':
    #     try:
    #         #if multiclass then convert to a single array, otherwise pass
    #         ska1=np.argmax(ska1,axis=1)
    #     except:
    #         pass
    #in some cases classification differences can be large, because we are dealing
    #with binary outcomes, so the max difference can be 1 for example
    # dif1=pd.Series(abs(ska1-skld))
    # print('difference between model and sklearn')
    # print(dif1.max())

    ##############################

    
    realizer=sentenceRealizerSymbolic()
    realizer.interpretSymbolic(res_symbolic,task=task)
    res_sentence=realizer.realizeAll()
    
    #This is the code for testing on the test data
    total_results_test=[]
    total_performance_test=[]
    if n_folds>0:
        if task=='classification':
            crossval=sklearn.model_selection.StratifiedKFold(n_splits=n_folds)
        else:
            crossval=sklearn.model_selection.KFold(n_splits=n_folds)
        df_reserve=df2.copy()
        g_reserve=copy.deepcopy(g)
        target_reserve=target.copy()
        for train, test in crossval.split(df2,target):

                df2_train=df2.iloc[train,:].copy()
                target_train=target[train].copy()
                # for i in range(len(g['best_features'])):
                #     g['best_features'][i]=g['best_features'][i][train]
                g2=g.copy()
                g2['best_all_feats_df']=g['best_all_feats_df'].iloc[train,:]
                
                
                df2_test=df2.iloc[test,:].copy()
                target_test=target[test]
                
                
                
                if not genetic_selection:
                    res_symbolic = findSymbolicExpression(df2_train,target,g2,scaler,task=task)
                else:
                    res_symbolic,final_features= findSymbolicExpression_genetic(df2_train,target_train,g2,scaler,task=task,
                                                                  ngen=ngen_for_second_round,
                                                                  population=npop_for_second_round,
                                                                  crossover_prob=crossover_second_round,
                                                                  mut_prob=mut_prob_second_round,
                                                                  individual_mut=individual_mut_second_round)
        

                #Clean up the results of the last function call findSymbolicExpression()
                model,final_models = tidyup_find_best(res_symbolic,g2,scaler,centerer,categorical_vars,filler,
                                                      task=task,target=target,
                                                      target_category_mapper=target_category_mapper,
                                                      numerical_variables=numerical_cols,
                                                      pca_object=pca_object,components=components)
                    
                results_test=model.evaluate(df2_test,fix_margins_low=fix_margins_low,
                                            fix_margins_high=fix_margins_high,ignore_pca=True)
                if task=='classification':
                    try:
                        #multiclass
                        performance_test=sklearn.metrics.cohen_kappa_score(target_test.tolist(),np.argmax(results_test,axis=1))
                    except:
                        performance_test=sklearn.metrics.cohen_kappa_score(target_test.tolist(),np.round(results_test).astype(np.int32))
                else:
                    #sometimes there are NaNs in the target
                    there_are_nans=np.where(np.isnan(target_test))[0]
                    # df_test3=df2_test.reset_index().drop(there_are_nans)
                    target_test=np.delete(target_test,there_are_nans)
                    results_test=np.delete(results_test.values,there_are_nans)
                    performance_test=sklearn.metrics.r2_score(target_test,results_test)
                    
                total_performance_test.append(performance_test)
                total_results_test.append(results_test)

    else:
        performance_test=None
        
    test_results=total_results_test
    performance_test=total_performance_test
    
    #fit the final model
    if not genetic_selection:
        res_symbolic = findSymbolicExpression(df2,target_reserve,g,scaler,task=task)
    else:
        res_symbolic,final_features= findSymbolicExpression_genetic(df2,target_reserve,g,scaler,task=task,
                                                      ngen=ngen_for_second_round,
                                                      population=npop_for_second_round,
                                                      crossover_prob=crossover_second_round,
                                                      mut_prob=mut_prob_second_round,
                                                      individual_mut=individual_mut_second_round)


    #Clean up the results of the last function call findSymbolicExpression()
    model,final_models = tidyup_find_best(res_symbolic,g,scaler,centerer,categorical_vars,filler,
                                          task=task,target=target,
                                          target_category_mapper=target_category_mapper,
                                          numerical_variables=numerical_cols,
                                          pca_object=pca_object,components=components)
    
    
    if bad_target_values is not None:
        train_results=model.evaluate(df.reset_index(drop=True).drop(bad_target_values),fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high,convert_class_output=False)
    else:
        train_results=model.evaluate(df2,fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high,convert_class_output=False)
        

        
    #Sometimes the model screws up and we have to circle around the other models
    dummy=model._sklearn_model.predict(final_features)
    successful_execution=_test_quality_results(train_results,model,task=task,
                                               target=dummy)    
    
    
    i=0
    while not successful_execution and i<=len(final_models):
        print('Initial model unsuccessful. Rotating other final models...')
        model2=final_models[i]

        if bad_target_values is not None:
            train_results=model2.evaluate(df.reset_index(drop=False).drop(bad_target_values),fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
        else:
            train_results=model2.evaluate(df,fix_margins_low=fix_margins_low,
                                    fix_margins_high=fix_margins_high)
            
        successful_execution=_test_quality_results(train_results,model2,task=task,
                                                   target=target)
        i+=1
        if i==len(final_models):
            print('ISSUES RUNNING MODELS. NO VALID MODEL FOUND.')
            break
        
        
    #Now that we chose a model, let's estimate the training performance
    if task=='classification':
        # for mapping in target_category_mapper.keys():
        #     target_test.values[target_test==mapping]=target_category_mapper[mapping]
        try:
            #multiclass
            performance_train=sklearn.metrics.cohen_kappa_score(target.tolist(),np.argmax(train_results,axis=1))
        except:
            performance_train=sklearn.metrics.cohen_kappa_score(target.tolist(),np.round(train_results).astype(np.int32))
    else:
        #sometimes there are NaNs in the target
        performance_train=sklearn.metrics.r2_score(target,train_results)    

    results={'model':model,'final_models':final_models,'issues_log':log,
             'natural_language_interpretation':res_sentence,'ground_truth_train':target,
             'ground_truth_test':target_test,
             'performance_test':performance_test,'successful_execution':successful_execution,
            'performance_train':performance_train,'predicted_train_values':train_results,
            'predicted_test_values':test_results,
            'components':components,'processed_input_data':df2,'sentence_realizer':realizer,
            'dataset_score':realizer.results['dataset_score']}
    
    return results
    
    
def explain_all_variables(model,df,target):
    results_log=""
     #Variable explanation
    #First we need to run this, as it will detect whether classification models are correct or not
    #If a classification model returns only the same value, then we shouldn't proceed with variable explanation
    #but rather urge the user to increase the total number of iterations
    
    #Shapley values explanations follows the following hierarchy
    
    #1. Ask the model to come up with shapley values
    shapley=model.explain(df)
    #2. Initialize an explainer
    shapley_realizer=variableExplainer()
    #target=df[target_name]
    #3. Create the basic statistics for shapley, which will be used to interpret results
    shapley_realizer.read_shapley(shapley,target)
    
    task=model._task
    
    #Wrong model can be, for example, returning the same class
    if not shapley_realizer.wrong_model:
        if task=='regression':
            columns=shapley.columns
        else:
            #Arbitrarily assume that class 0 is the important one. Hence, even in
            #multiclass classification problems, class 0 becomes the reference class
            columns=shapley[0].columns
            #regression
        for col in columns:
            shapley_realizer.explainSingleVariable(col)
            #General function to summarise results
            res_shapley=shapley_realizer.realizeAll()
            print(res_shapley)
            print('\n')
            results_log+='\n Variable importance results for "'+col+ '":\n'+res_shapley+"\n"
        
        
        
        #This shapley class reads all variables at once
        shapley_summary = allVariablesExplainer()
        #find the patterns in the variables
        shapley_summary.read_shapley(shapley,target)
        #convert to natural language
        shapley_summary.explainVariablesSummary()
        res_shapley_summary=shapley_summary.realizeAll()
        print(res_shapley_summary)
        results_log+='\n'+res_shapley_summary
        
        #most important variables for this model
        important_variables = shapley_summary.getImportantVariableNames()
        components_res=model.explain_components_from_variables(important_variables)
        
        
        importance_breakdown=model.explain_new_data(df)

        results={'most_important_variables':important_variables,
                 'log':results_log,
                 'shapley_summary':res_shapley_summary,
                 'shapley_summary_objects':shapley_summary,
                 'components_explanation':components_res,
                 'detailed_breakdown':importance_breakdown}
        
    else:
        results={'most_important_variables':None,
         'log':None,
         'shapley_summary':None,
         'shapley_summary_objects':None,
         'components_explanation':None,
         'detailed_breakdown':None}
    return results

def run_pattern_extraction(df):
    #AUTOMATED INTELLIGENT PATTERN DETECTION
    rules=associationRuleMining(df)
    print(rules)
    
    cors=detectHighCorrelations(df)
    print(cors)
    
    #both of these are dataframe objects
    results={'rules':rules,'correlations':cors}
    
    return results
    
def run_causal_analysis(results_em,variable_importance_results,target,vars_to_exclude=[],
                        variable_size_limit=10):
    """
    vars_to_exclude: In some cases, you want to simply control for some variables, but you
    do not believe there is a causal relationship (or the causal effect doesn't make sense.)
    """
    variables=results_em['model'].return_variables()
    #if the variabls of the model are too many, then simply select the ones with above median
    #importance
    if len(variables)>variable_size_limit:        
        variables=variable_importance_results['shapley_summary_objects'].getImportantVariableNames(quant=0.5)
    
    for var in vars_to_exclude:
        variables=np.delete(variables,np.argwhere(variables==var))
    
    causal_res=run_causality(results_em['processed_input_data'],results_em['ground_truth_train'],
                             important_variables=variables)
    causal_results_inter,_=interpret_causal_results(causal_res)
    print(causal_results_inter)
    #causal interpretation is natural language explanation of the raw results 
    results={'causal_results_interpretation':causal_results_inter,'causal_results':causal_res}
    return results
    
def infer_task(target):
    target=target.copy()
    target = target[np.logical_not(np.isnan(target))]
    
    if target[0]==str:
        return 'classification'

    if len(np.unique(target))<=5:
        return 'classification'
    
    return 'regression'

def perform_ml_test(df,target_name,task,fix_skew):    
    
    df2, target,scaler,centerer,categorical_vars,filler,log,\
    target_category_mapper,numerical_cols,components,pca_object,bad_target_values=\
    prepareTrainData(dataframe=df,target_name=target_name,task=task,fix_skew=fix_skew,createFeats=False)
    
    df2=pd.get_dummies(df2)
    
    if task=='regression':
        model=LGBMRegressor(n_estimators=df2.shape[1])
        metric='r2'
    else:
        model=LGBMClassifier(n_estimators=df2.shape[1])
        metric=sklearn.metrics.make_scorer(sklearn.metrics.cohen_kappa_score)
    
    score=sklearn.model_selection.cross_val_score(model,df2,target,scoring=metric,cv=5)
    return score
    

def run_automl_models(df,target_name,task,fix_skew=False,num_iterations=20,reference_class=1):    
    
    df2, target,scaler,centerer,categorical_vars,filler,log,\
    target_category_mapper,numerical_cols,components,pca_object,bad_target_values=\
    prepareTrainData(dataframe=df,target_name=target_name,
                     task=task,fix_skew=fix_skew,createFeats=False,pca_criterion_thres=np.inf)
    
    df2=pd.get_dummies(df2)
    
    if task=='regression':
        model=LGBMRegressor(n_estimators=df2.shape[1])
        metric='r2'
    else:
        model=LGBMClassifier(n_estimators=df2.shape[1])
        metric='cohen_kappa'
    
    
    
    n_estimators=scipy.stats.randint(10,max(df.shape[1],30))
    max_depth=scipy.stats.randint(1,min(df.shape[1],30))
    lambda_l1=np.arange(0,0.8,0.05)
    lambda_l2=np.arange(0,0.8,0.05)
    extra_trees=[True,False]
    subsample=np.arange(0.2,1.025,0.025)
    bagging_freq=scipy.stats.randint(1,200)
    colsample_bytree=np.arange(0.3,1.025,0.025)
    num_leaves=scipy.stats.randint(1,100)
    boosting=['gbdt','dart','rf']
    drop_rate=np.arange(0.1,0.8,0.025)
    skip_drop=np.arange(0,0.8,0.025)
    alpha=scipy.stats.randint(10,2000)
    learning_rate=[0.00001,0.0001,0.001,0.01,0.015,0.02]+np.arange(0.02,0.555,0.005).tolist()
    
    
    params = {'n_estimators':n_estimators,'max_depth':max_depth, 'subsample':subsample,
              'colsample_bytree':colsample_bytree,
              'learning_rate':learning_rate,
              'num_leaves':num_leaves,
              'boosting':boosting,
              'extra_trees':extra_trees,
              'lambda_l1':lambda_l1,
              'lambda_l2':lambda_l2,
              'bagging_freq':bagging_freq,
              'drop_rate':drop_rate,
              'alpha':alpha}
    
    
    params = list(ParameterSampler(params, n_iter=num_iterations))
    
    results=[]

    for param in params:
        print('iterating...')
        if task=='regression':
            model=LGBMRegressor(**param)
            metric='r2'
        else:
            model=LGBMClassifier(**param)
            metric='roc_auc'
        score=np.mean(sklearn.model_selection.cross_val_score(model,df2,target,scoring=metric,cv=3))
        results.append({'param':param,'metric':score})
    
    results=pd.DataFrame(results)
    results.sort_values('metric',ascending=False,inplace=True)
    
    #fit final model ater you get the CV score
    if task=='regression':
        model=LGBMRegressor(**param)
        metric='r2'
    else:
        model=LGBMClassifier(**param)
    
    model.fit(df2,target)
    
    #feature importance
    feats=model.feature_importances_
    cols=df2.columns
    
    features=pd.Series(feats,index=cols)
    features=features.sort_values(ascending=False)
    features=features/features.sum()
    
    
    if task=='regression':
        tree=DecisionTreeRegressor()
    else:
        tree=DecisionTreeClassifier()

    
    #shapley values
    tree.fit(df2,model.predict(df2))
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(df2)
    
    if task=='regression':
        dfs=pd.DataFrame(shap_values,columns=features.index)
    else:
        dfs=[]
        for i in range(len(shap_values)):
            dfs.append(pd.DataFrame(shap_values[i],columns=cols))
        #d = reduce(lambda x, y: x.add(y, fill_value=0), dfs)
       # dfs = d/len(dfs)
        dfs=dfs[reference_class]

    values=dfs
    percentages=values.abs().sum()/values.abs().sum().sum()
    average_contribution=values.mean()
    
    values[target_name]=target
    values.columns=values.columns.str.replace('var_','')
    
    predictions=model.predict(df2)
    
    return {'percentage_contribution':percentages,'expected_mean_contribution':average_contribution,
            'full_values':values,'metric':np.round(results['metric'].values[0],3),'predictions':predictions}  


def get_importance_histogram(feats_dict,original_df,variable):
    full_values=feats_dict['full_values']
    var1=full_values[variable]
    var2=original_df[variable]
    
    df=pd.DataFrame({'impact':var1,variable:var2})
    
    return df
    
        
    

def run_all(path_or_data,target_name,task='infer',quant_or_num=0.6,ngen=20,sample_n=-1,max_tree=3,
             n_pop=[50,50,50,50],n_features=3,store_results='results_tests.csv',allowed_time=6,
             test_perc=0.15,choice_method='quant',fix_skew=True,limit_n_for_feature_selection=-1,
             target_sampling=0.9,pca_criterion_thres=0.05,selection_variable_ratio=0.1,
             extract_patterns=True,causal=False,ngen_for_second_round=10,npop_for_second_round=10,
             crossover_second_round=0.5,
             mut_prob_second_round=0.1,individual_mut_second_round=0.1,fix_margins_low=True,
                                    fix_margins_high=True,dtype=None,n_folds=-1,variables_to_keep=None,
                                    perform_ml_benchmark=True):
    """
    path_or_data:Either a string path to a .csv/Excel or a pandas dataframe
    
    task: this can be either 'regression','classification' or 'infer' (which infers the task automatically)
        
    choice_method: The method of choice for choosing thebest features. Choices are 'num' 
    and 'quant'. Num means that only the best X features will be chosen. Quant means that 
    features with score>quantile will be chosen. E.g. choosing quant_or_num=0.5 will keep
    features with above median score.
    
    quant_or_num: related to choice method
    
    ngen: The number of generations for the genetic algorithm
    
    max_tree: The maximum depth of the tree used by the genetic programming algorithm
    
    n_pop: The population. This taks a list because we are using the islands version of genetic algorithms
    
    n_features: The final number of features to ue for the equation
    
    store_results: Where to save the results
    
    allowed_time: The allowed execution time of the genetic algorithm in minutes. The actual
    running time might be slightly longer due to some additional processing that need to be done.
    
    test_perc: ADAN can perform a train/test split. The determines the % for the test split.
    
    fix_skew: If True, then ADAN will test to see whether the data is skewed. If yes
    
    sample_n: If -1 then all the rows of the dataset are used. You get this percentage. This used to 
    be an integer value, but now the name has been kept, but it is a %

    
    limit_n_for_feature_selection: How many rows to use when selecting features. We need to take 
    a sample, otherwise it takes a very long time. This is a very important parameter. Ideally we want
    to use as a big of a sample as possible. If the parameter is set to -1, then we are taking 
    (by default) all features
    
    
    pca_criterion_threshold: Calculate correlations between features. Get the mean and the std.
    If mean-std>threshold, then run PCA.
    
    target_sampling: When the features are evaluated, we can sample a % of the targets, and evaluate
    the performace on this subset. This should help with overfitting and finding better solutions.
    
    selection_variable_threshold: This is defined as (number of variables)/(number of rows). If this
    ratio is above the threshold, then perform feature selection. Feature selection can be very expensive, 
    so, it is not worth doing when there are many more rows than columns.
    
    causal: Whether to run causal analysis or not
    
    extract_patterns: Whether to extract patterns or not. This is done through the use of frequent item
    mining algorithms, correlations and other heuristics.
    
    ***these variables are used for the genetic algorithm in run_Equatin_model step 2***
    ngen_for_second_round: *
    npop_for_second_round: *
    crossover_second_round: *
    mut_prob_second_round: *
    individual_mut_second_round: if a mutation has been decided to occur, 
    then each bit of the feature vector [0,1,1,1,0,...,1] will be mutated according to this probability
    
    fix_margins_low/high: This variable refers to whether the predictions should go
    below or above the minimum/maximum target value in original dataset (e.g. some values
    can't be below 0). It makes sense to turn it to True in some contexts, but this requires domain
    knowledge. In general, setting it to True is a safe choice, unless we care about extrapolation.
    
    dtype: The type of the columns (if provided by the user)
    
    n_folds: The number of folds for cross-validation. If n_folds>2, then
    cross validation overrides train/test split
    
    variables_to_keep: If None, then keep all, otherwise keep only the ones in the list
    
    """

    
    results_log=""
    res_shapley_summary=[]
    rules=[]
    cors=[]
    causal_results=[]
    important_variables=[]
    causal_results_inter=[]
    

    df,log_read_data=readData(path_or_data,dtype=dtype)
    
    
    if df.shape[0]>sample_n and sample_n>1:
        tosample=int(sample_n*df.shape[0])
        if(tosample<10):
            print('the sample is too small, proceeding with the full dataset')
            tosample=10
        df=df.sample(tosample)
    df.index=np.arange(0,df.shape[0])
    
    if variables_to_keep is not None:
        df=df.loc[:,variables_to_keep]
    
    if task=='infer':
        task=infer_task(df[target_name])

    log_transform_applied=False
    #transform the target if it is too skewed
    if task=='regression' and fix_skew:
        try:
            df[target_name]=pd.to_numeric(df[target_name])
        except:
            df[target_name]=pd.to_numeric(df[target_name],errors='coerce')
            print('WARNING: Since the task it is a regression, the target variable will be converted to numerical type. \
                  Some cells will be removed.')
        target=df[target_name].dropna()
        if np.abs(sp.stats.skew(target))>2.5:
            print('EXCESSIVE SKEWNESS DETECTED. USING A LOGARITHM TO FIX IT.')
            if(all(df[target_name]>0)):
                target2=np.log(target+1)
            else:
                target2=np.log(df[target_name]+abs(min(df[target_name]))+1)
            log_transform_applied=True
            df[target_name]=target2
    else:
        log_transform_applied=False
        
    if limit_n_for_feature_selection==-1:
        limit_n_for_feature_selection=df.shape[0]
    
   
    
    #NOTE: Variable importance requires one to first run equation modelling.
    # Causal results require one to run both equation modelling and variable importance
    #Pattern extraction has no pre-requisites
    
    #holds the best model, and all other models found through pareto optimality criterion
    print('running equation model')
    try:
        if n_folds<3:
            results_em=run_equation_model(df=df,target_name=target_name,task=task,
                                          quant_or_num=quant_or_num,ngen=ngen,max_tree=max_tree,
                                          n_pop=n_pop,n_features=n_features,
                                          allowed_time=allowed_time,fix_skew=fix_skew,test_perc=test_perc,
                                          limit_n_for_feature_selection=limit_n_for_feature_selection,
                                          target_sampling=target_sampling,ngen_for_second_round=ngen_for_second_round,
                                          npop_for_second_round=npop_for_second_round,
                                          crossover_second_round=crossover_second_round,
                     mut_prob_second_round=mut_prob_second_round,
                     individual_mut_second_round=individual_mut_second_round,
                     selection_variable_ratio=selection_variable_ratio,
                     fix_margins_low=fix_margins_low,fix_margins_high=fix_margins_high,
                     pca_criterion_thres=pca_criterion_thres)
        else:
            results_em=run_equation_model_cv(df=df,target_name=target_name,task=task,
                                          quant_or_num=quant_or_num,ngen=ngen,max_tree=max_tree,
                                          n_pop=n_pop,n_features=n_features,
                                          allowed_time=allowed_time,fix_skew=fix_skew,test_perc=test_perc,
                                          limit_n_for_feature_selection=limit_n_for_feature_selection,
                                          target_sampling=target_sampling,ngen_for_second_round=ngen_for_second_round,
                                          npop_for_second_round=npop_for_second_round,
                                          crossover_second_round=crossover_second_round,
                     mut_prob_second_round=mut_prob_second_round,
                     individual_mut_second_round=individual_mut_second_round,
                     selection_variable_ratio=selection_variable_ratio,
                     fix_margins_low=fix_margins_low,fix_margins_high=fix_margins_high,n_folds=n_folds,
                     pca_criterion_thres=pca_criterion_thres)
    except:
        results_em={'successful_execution':False}
    
    if results_em['successful_execution']==False:
        print('Issues with execution, please re-run')
        return None
    #returns the 3 most important variables, a description of the importance of each variable, and a shapley summary object
    #to be used in the causality module
    print('running variable importance')
    variable_importance_results=explain_all_variables(model=results_em['model'],
                                                      df=df,
                                                      target=results_em['predicted_train_values'])
    
    #Get general patterns. This is restricted now to rules (frequent itemset mining) and correlations
    if extract_patterns:
        print('running pattern extraction')
        pattern_extraction=run_pattern_extraction(df)
    else:
        pattern_extraction={'rules':None,'correlations':None}
    #returns an interpretation of the causal results and the raw results
    if causal and variable_importance_results is not None:
        print('running causal inference')
        causal_results = run_causal_analysis(results_em,variable_importance_results,results_em['ground_truth_train'])
    else:
        causal_results={'causal_results_interpretation':None,'causal_results':None}
    constraints=_find_constraints(df)
    
    
    #Normally this should go into the interpretability module
    comparison=None
    score=None
    if perform_ml_benchmark and results_em['performance_test'] is not None:
        score=perform_ml_test(df,target_name,task,fix_skew=fix_skew)
        ratio,comparison,can_trust_test,use_test=interpret_generalisation(results_em,score)
    else:
        ratio=None
        comparison=None
        can_trust_test=None
        use_test=None
        
        
    
    results_dict={
            'model':results_em['model'],
            'other_models':results_em['final_models'],
            'most_important_variables':variable_importance_results['most_important_variables'],
            'detailed_variable_importance_breakdown':variable_importance_results['detailed_breakdown'],
            'variable_importance_natural_language':variable_importance_results['log'],
            'general_patterns_rules':pattern_extraction['rules'],
            'general_patterns_correlations':pattern_extraction['correlations'],
            'causal_results_natural_language':causal_results['causal_results_interpretation']   ,
            'model_interpretation_natural_language':results_em['natural_language_interpretation'],
            'performance_test':results_em['performance_test'],
            'log_transform_applied_to_target':log_transform_applied,
            'performance_train':results_em['performance_train'],
            'shapley_summary':variable_importance_results['shapley_summary'],
            'processed_input_data':results_em['processed_input_data'],
            'variable_importance_results_dict':variable_importance_results,
            'results_em_dict':results_em,
            'successful_execution':results_em['successful_execution'],
            'predicted_train_values':results_em['predicted_train_values'],
            'predicted_test_values':results_em['predicted_test_values'],
            'pca_components':results_em['components'],
            'ground_truth_train':results_em['ground_truth_train'],
            'ground_truth_test':results_em['ground_truth_test'],
            'issues_log':results_em['issues_log'],
            'natural_language_PCA':variable_importance_results['components_explanation'],
            'realizer':results_em['sentence_realizer'],
            'constraints':constraints,
            'issues_log':results_em['issues_log'],
            'comparison_ml':comparison,
            'score_ml':score,
            'ml_vs_adan':ratio,
            'use_test':use_test
            }
            
    return results_dict
    
#Test script
# results=run_all('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/iris/iris.csv',
#                   target_name=4,task='classification') 
# fake_df={0:1.54,1:2.23,2:4.56,3:1.25}
# model=results['model']
# new_res=model.evaluate(fake_df)
# new_res=model.evaluate(fake_df,True)
  
#print(results['natural_language_PCA'])
# results=run_all('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/abalone/abalone.csv',
#                 target_name=0,task='classification')   
# model=results['model']

# results=run_all('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/auto_mpg/auto_mpg.csv',
#                 target_name='V1',task='regression')  
# model=results['model']


# results=run_all('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/breast_cancer/breast_cancer.csv',
#                 target_name='class',task='classification')  
# model=results['model']

# results=run_all('/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/london_crime/london_crime.csv',
#                 target_name='value',task='regression')   
# print(results['natural_language_PCA'])
