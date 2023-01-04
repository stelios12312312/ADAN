#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:50:02 2019

@author: stelios
"""

import os, sys
lib_path = os.path.abspath(os.path.join('../..'))+"/adan"
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../../'))+"/adan"
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('..'))+"/adan"
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

lib_path = os.path.abspath(os.path.join('../..'))
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


# np.random.seed(111)

path="data_processed.csv"

london_crime=run_all(target_name='Price',ngen=40,task='regression',
                    path_or_data=path,max_tree=5,
                      sample_n=-1,choice_method='quant',quant_or_num=1,
                      fix_skew=False,allowed_time=60*5,n_pop=[20,20,20,20],
                      test_perc=0.0,limit_n_for_feature_selection=19000,n_features=15,
                      target_sampling=0.9,pca_criterion_thres=0.25,fix_margins_low=True,
                      fix_margins_high=True)
# london_crime=run_all(target_name='value',task='regression',
#                      path_or_data=path,n_pop=[50,50,50,50])


print('****CAUSAL INFERENCE***')
print(london_crime['causal_results_natural_language'])
print('****PERFORMANCE***')
print(london_crime['performance_train'])
print('****VARIABLE IMPORTANCE***')
print(london_crime['shapley_summary'])
print('****MODEL******')
print(london_crime['model'].model)
#print(london_crime['pca_components'])
#from matplotlib import pyplot as plt
error = np.mean(np.abs(london_crime['predicted_train_values']-london_crime['ground_truth_train']))
print(error)
plt.close()
plt.scatter(london_crime['predicted_train_values'],london_crime['ground_truth_train'])
plt.xlabel('predicted')
plt.ylabel('true values')

error = np.mean(np.abs(london_crime['predicted_test_values']-london_crime['ground_truth_test']))
plt.scatter(london_crime['predicted_test_values'],london_crime['ground_truth_test'])
print(error)
df=pd.read_csv("data_processed.csv")

preds=london_crime['model'].evaluate(df)
# plt.scatter(preds,df['value'])

