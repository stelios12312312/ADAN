#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 15:03:37 2021

@author: stelios
"""
import pandas as pd
import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from adan.protocols import *
from adan.aipm import main_predictive

df=pd.read_csv("auto_mpg/auto_mpg.csv")
target_name='V1'
task='regression'


# feats=run_automl_models(df,'V1','regression',num_iterations=1)
# get_importance_histogram(feats,df,'V3')

# model=run_all(df,'V1','regression',n_pop=[10,10],allowed_time=3)


# preds1=feats['predictions']
# preds2=model['model'].evaluate(df)
# np.corrcoef(preds1,preds2)[0][1]

#Deep automl function
df=pd.read_csv("auto_mpg/auto_mpg.csv")
target_name='V1'
task='regression'
df2, target,scaler,centerer,categorical_vars,filler,log,\
target_category_mapper,numerical_cols,components,pca_object,bad_target_values=\
prepareTrainData(dataframe=df,target_name=target_name,task=task)
              
automl=main_predictive.predictor_main(df2,target)
results=automl.train_evaluate_models(optimizer_type='hypersearch',optimizer_params={},models='lightweight')