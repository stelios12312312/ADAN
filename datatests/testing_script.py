#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:37:44 2021

@author: stelios
"""

import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from datetime import datetime

import cdt
import adan
from adan.aiem.genetics import *
from adan.aidc.utilities import *
from adan.aidc.feature_selection import *


from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers.hyperparam_optimization import *

from adan.aiem.symbolic_modelling import *
from adan.aiem.genetics.genetic_programming import *
from adan.protocols import *
from matplotlib import pyplot as plt
from adan.aist.mappers import *
from adan.aipd.aipd_main import *
from adan.aiem.symbolic_modelling import *
from adan.aiem.optimise import reverse_optimise
import os
import sklearn

import random
import numpy as np
import pandas as pd

"""
https://data.london.gov.uk/dataset/estimation-health-impacts-particulate-pollution-london

This is an air pollution dataset used for demo purposes.

We need to remove 2 variables, because they are potential target variables.
"""

n_pop=[20,20,20]
path="particulate_air_pollution_mortality/particulate-air-pollution-mortality (1).csv"
df=readData(path)
df=pd.read_csv(path,encoding='latin1')
df.drop(['Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 6%',
       'Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 1%'],axis=1,inplace=True)

for i in range(3):
    print(i)
    print('*'*1000)
    path_or_data = df
    target_name = 'Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 12%'
    task='regression'
    quant_or_num=round(random.uniform(0.1, 0.9), 1)
    ngen = random.randint(1, 50)
    sample_n = -1
    max_tree = random.randint(2, 7)
    n_pop = [50]*random.randint(1, 10)
    n_features = "based on value"
    allowed_time=6
    test_perc=0.15
    choice_method='quant'
    fix_skew=True
    limit_n_for_feature_selection=-1
    target_sampling=round(random.uniform(0.1, 0.9), 1)
    pca_criterion_thres=0.05
    selection_variable_ratio=round(random.uniform(0.1, 0.9), 1)
    extract_patterns=False
    causal=False
    ngen_for_second_round=10
    npop_for_second_round=10
    crossover_second_round=0.5
    mut_prob_second_round=0.1
    individual_mut_second_round=0.1
    fix_margins_low=True
    fix_margins_high=True
    dtype=None
    n_folds=-1
    variables_to_keep=None
    store_results='results_tests.csv'

    result = run_all(path_or_data = path_or_data,target_name = target_name,task=task,quant_or_num=quant_or_num,ngen = ngen,
    sample_n =sample_n,max_tree = max_tree, n_pop = n_pop, n_features = n_features,
    allowed_time=allowed_time,test_perc=test_perc,choice_method=choice_method,fix_skew=fix_skew,limit_n_for_feature_selection=limit_n_for_feature_selection,
    target_sampling=target_sampling,pca_criterion_thres=pca_criterion_thres,
    selection_variable_ratio=selection_variable_ratio,
    extract_patterns=extract_patterns,
    causal=causal,
    ngen_for_second_round=ngen_for_second_round,
    npop_for_second_round=npop_for_second_round,
    crossover_second_round=crossover_second_round,
    mut_prob_second_round=mut_prob_second_round,
    individual_mut_second_round=individual_mut_second_round,
    fix_margins_low=fix_margins_low,
    fix_margins_high=fix_margins_high,
    dtype=dtype,
    n_folds=n_folds,
    variables_to_keep=variables_to_keep,
    store_results=store_results)
    
    print('*'*1000)


    parameters = {
    "path_or_data" : path_or_data,
    "target_name" : target_name,
    "task" : task,
    "quant_or_num":quant_or_num,
    "ngen" : ngen,
    "sample_n" :sample_n,
    "max_tree" : max_tree,
    "n_pop" : (50, len(n_pop)),
    "n_features" : n_features,
    "allowed_time":allowed_time,
    "test_perc":test_perc,
    "choice_method":choice_method,
    "fix_skew":fix_skew,
    "limit_n_for_feature_selection":limit_n_for_feature_selection,
    "target_sampling":target_sampling,
    "pca_criterion_thres":pca_criterion_thres,
    "selection_variable_ratio":selection_variable_ratio,
    "extract_patterns":extract_patterns,
    "causal":causal,
    "ngen_for_second_round":ngen_for_second_round,
    "npop_for_second_round":npop_for_second_round,
    "crossover_second_round":crossover_second_round,
    "mut_prob_second_round":mut_prob_second_round,
    "individual_mut_second_round":individual_mut_second_round,
    "fix_margins_low":fix_margins_low,
    "fix_margins_high":fix_margins_high,
    "dtype":dtype,
    "n_folds":n_folds,
    "variables_to_keep":variables_to_keep,
    "store_results":store_results,
    "result" :result
    }

    
    try:
        data = pd.read_csv('results.csv').iloc[:,1:]
        data = data.append(parameters, ignore_index=True)
    except:
        pass
    data.to_csv('results.csv')
    
    

    
