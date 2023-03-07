#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:44:55 2022

@author: stelios
"""
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)


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

import random
#random.seed(27)

df=pd.read_csv('data/churn_data/churn.csv')
df=df.sample(frac=0.2)

cats=['product_id','postcode','partner_name','merchant_name','manufacturer','model',
      'product_coverage', 'contract_type_group','device_category','condition','color']

for cat in cats:
    df[cat]=df[cat].astype('category')


feats=run_automl_models(df,'cancel_target','classification',num_iterations=3)

#paok=run_all(df,'cancel_target',n_folds=-1,test_perc=0.05,task='classification',allowed_time=50)


# import autosklearn.classification
# cls = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=60)

# X_train=df.drop('cancel_target',axis=1)
# y_train=df['cancel_target']
# cls.fit(X_train, y_train)
#predictions = cls.predict(X_test)