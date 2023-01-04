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

"""
https://data.london.gov.uk/dataset/estimation-health-impacts-particulate-pollution-london

This is an air pollution dataset used for demo purposes.

We need to remove 2 variables, because they are potential target variables.
"""

path="data/particulate_air_pollution_mortality/particulate-air-pollution-mortality (1).csv"
df=pd.read_csv(path,encoding='latin1')
df.drop(['Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 6%',
       'Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 1%'],axis=1,inplace=True)


mortality = run_all(target_name='Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 12%',\
                      ngen=10,quant_or_num=0.9,task='regression',path_or_data=df,\
                          sample_n=-1,n_pop=[20,20],test_perc=0.0,ngen_for_second_round=2,n_folds=-1)
    
plt.scatter(mortality['predicted_train_values'],mortality['ground_truth_train'])

print(str(mortality['model'].model))
    

    