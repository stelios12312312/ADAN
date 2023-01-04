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
import cdt
cdt.SETTINGS.rpath='/usr/local/bin/Rscript'

def concordance(x,y):
    sx=np.std(x)
    sy=np.std(y)
    meanx=np.mean(x)
    meany=np.mean(y)
    cor=np.corrcoef(x,y)[0][1]
    
    numerator=2*cor*sx*sy
    denom=sx**2+sy**2+(meanx-meany)**2
    
    return numerator/denom

# np.random.seed(111)
path="fundamentals_ADAN.csv"
price_threshold=1000
npop=[50,50,50,50]
target_sampling=0.25
max_tree=20
ngen=15

#the data is in chronological order. The first X points are used for training
train_data_perc=0.9

industry='Trading'
data=pd.read_csv(path)
data=data[data['famaindustry']==industry]
data.drop(['sicsector','famaindustry','Unnamed: 0','calendardate','ticker'],axis=1,inplace=True)

#Remove penny stocks
data=data[data['price']>1]
data=data[data['price']<price_threshold]
# data=data.sample(1000)
print(data.shape)
starttest=int(data.shape[0]*train_data_perc)
testtime=data.iloc[starttest:,:]
data=data.iloc[:starttest,:]

#main functions
london_crime=run_all(target_name='price',ngen=ngen,task='regression',
                    path_or_data=data,max_tree=max_tree,
                      sample_n=-1,choice_method='quant',quant_or_num=0.25,
                      fix_skew=False,allowed_time=3000,n_pop=npop,
                      test_perc=0.15,limit_n_for_feature_selection=int(data.shape[0]/1.5),n_features=150,
                      target_sampling=target_sampling,pca_criterion_thres=0.05,
                      ngen_for_second_round=10,npop_for_second_round=10,fix_margins_low=True)



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

print('performance test')
print(sklearn.metrics.r2_score(london_crime['ground_truth_test'],london_crime['predicted_test_values']))
print('performance train')
print(sklearn.metrics.r2_score(london_crime['ground_truth_train'],london_crime['predicted_train_values']))
print(np.corrcoef(london_crime['predicted_test_values'],london_crime['ground_truth_test']))
print('concordance')
print(concordance(london_crime['predicted_test_values'],london_crime['ground_truth_test']))

print('New preds')
preds=london_crime['model'].evaluate(testtime,fix_margins_low=True)
print(sklearn.metrics.r2_score(testtime.price,preds))
print(np.corrcoef(testtime.price,preds))
print(concordance(testtime.price,preds))


# plt.scatter(np.exp(london_crime['predicted_test_values']),np.exp(london_crime['ground_truth_test']))

# print('TRANSFORMED METRICS')
# print(np.mean(abs(np.exp(london_crime['predicted_train_values'])-np.exp(london_crime['ground_truth_train']))))
# print(np.mean(abs(np.exp(london_crime['predicted_test_values'])-np.exp(london_crime['ground_truth_test']))))
# print(sklearn.metrics.r2_score(np.exp(london_crime['ground_truth_test']),np.exp(london_crime['predicted_test_values'])))
# print(concordance(np.exp(london_crime['predicted_test_values']),np.exp(london_crime['ground_truth_test'])))

# df=pd.read_csv("/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/datatests/london_crime/london_crime.csv")

# preds=london_crime['model'].evaluate(df)
# plt.scatter(preds,df['value'])
