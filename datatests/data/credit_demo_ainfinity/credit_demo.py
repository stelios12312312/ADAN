#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:11:17 2020

@author: stelios
"""


import os, sys
lib_path = os.path.abspath(os.path.join('..'))+"/adan"
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)


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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix,classification_report

import cdt
cdt.SETTINGS.rpath='/usr/local/bin/Rscript'

# np.random.seed(111)

path="credit_train.csv"

#BASIC FUNCTION, play around with this
# london_crime=run_all(target_name='SeriousDlqin2yrs',task='classification',
#                      path_or_data=path,n_pop=[50,50,50,50],ngen=20,max_tree=3,n_features=5,
#                      quant_or_num=0.6,limit_n_for_feature_selection=15000,
#                      allowed_time=1000)

#Shivamm 0.27
london_crime=run_all(target_name='SeriousDlqin2yrs',task='classification',
                     path_or_data=path,n_pop=[50,50,50,50],ngen=100,max_tree=7,n_features=10,
                     quant_or_num=0.7,limit_n_for_feature_selection=23000,
                     allowed_time=300)


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
kappa=cohen_kappa_score(np.round(london_crime['predicted_train_values']),london_crime['ground_truth_train'])
kappa_test=cohen_kappa_score(np.round(london_crime['predicted_test_values']),london_crime['ground_truth_test'])
conf=classification_report(np.round(london_crime['predicted_test_values']),london_crime['ground_truth_test'])
print(kappa)
print(kappa_test)
print(conf)

#TRY with different threshold, doesn't seem to work very well
threshold=0.5
preds=london_crime['predicted_test_values']
preds2=preds.copy()
preds2[preds2>threshold]=1
preds2[preds2<=threshold]=0
conf=classification_report(preds2,london_crime['ground_truth_test'])
print(conf)


df=pd.read_csv("credit_train.csv")

preds=london_crime['model'].evaluate(df)
