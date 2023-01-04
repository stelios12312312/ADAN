#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 11:38:54 2018

@author: stelios
"""

import os, sys
lib_path = os.path.abspath(os.path.join('..'))+"/adan"
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

import adan
from adan.aiem.genetics import *
from adan.aidc.utilities import *
from adan.aidc.feature_selection import *
#from adan.aipm.optimizers.xgboost_wrapper import *
#from adan.aipm.optimizers.optimizers import *
#from adan.aipm.optimizers.xgbOptimizer import *

from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers.hyperparam_optimization import *

from adan.aiem.symbolic_modelling import *
from adan.aist.mappers import *
from adan.aidc.utilities import *
#from adan.aipm.optimizers import *
from adan.aidc.utilities import *
from adan.aiem.genetics.genetic_programming import *
#from adan.modellingDeprecated.estimation_utilities import *
#from adan.aipm.optimizers.xgbOptimizer import *
from matplotlib import pyplot as plt
from adan.protocols import *


np.random.seed(111)

print('Community crimes')
#https://www.kaggle.com/kkanda/analyzing-uci-crime-and-communities-dataset/data

target_name='ViolentCrimesPerPop'
path="/community crimes/crimedata_removed_columns.csv"
task='regression'
ngen=10

df=readData(os.path.dirname(os.path.abspath(__file__))+path)
#df.drop(['communityname','state','assaultPerPop','larcPerPop'],inplace=True,axis=1)
#df.drop(['communityname','state','murdPerPop','rapesPerPop','robbbPerPop','assaultPerPop',
#         'rapes','assaults','robberies','murders','autoTheft','autoTheftPerPop','burglPerPop',
#         'nonViolPerPop'],inplace=True,axis=1)
#this is where the cleaning takes place
#this function
#crime=run_all(target_name='ViolentCrimesPerPop',ngen=50,task='regression',
#                     path_or_data=df,
#                       sample_n=-1,choice_method='quant',quant_or_num=0.5,
#                       fix_skew=True,allowed_time=5,n_pop=[50,50,50,50],
#                       test_perc=0.1,limit_n_for_feature_selection=1500,n_features=2,
#                       target_sampling=0.7,pca_criterion_thres=0.05)

crime=run_all(target_name='ViolentCrimesPerPop',task='regression',
                     path_or_data=df[0],n_pop=[50,50,50,50])
print('****CAUSAL INFERENCE***')
print(crime['causal_results_natural_language'])
print('****PERFORMANCE***')
print(crime['performance_train'])
print('****VARIABLE IMPORTANCE***')
print(crime['shapley_summary'])
print('****MODEL******')
print(crime['model'].model)
preds=crime['model'].evaluate(df)


error = np.mean(np.abs(crime['predicted_train_values']-crime['ground_truth_train']))
print(error)
plt.close()
plt.scatter(crime['predicted_train_values'],crime['ground_truth_train'])
plt.xlabel('predicted')
plt.ylabel('true values')

error = np.mean(np.abs(crime['predicted_test_values']-crime['ground_truth_test']))
print(error)
