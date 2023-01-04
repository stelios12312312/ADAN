# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
#from feature_preprocess import *

#import os, sys
#lib_path = os.path.abspath(os.path.join('..','..'))
#sys.path.append(lib_path)
import adan


#from adan.functions import *

import pandas as pd
import itertools
import time
import multiprocessing
from typing import Callable, Tuple, Union

# from pandarallel import pandarallel
 

def apply_all(tup,functions):
    colname=tup[0]
    group=tup[1]
    group.drop(group.columns[0],inplace=True,axis=1)
    # res=[]
    # for func in functions:
    #     res.append(group.apply(func))
    # res=pd.concat(res)
    res=group.agg(functions[0:3])
    return res

def aggregatePerColumn(functions,df,parallel=False):
    
    if parallel:
        pandarallel.initialize()
    
    if np.all(df.dtypes=='category'):
        return df
    df=df.copy()
    columns=df.select_dtypes(include=['category']).columns
    groups=[]
    for column in columns:
#        try:
#            g=df.groupby(by=column,as_index=False).agg(functions)
#            g.columns = ['_'.join(col).strip()+"_over_"+column for col in g.columns.values]
#            g[column]=g.index
#            groups.append(g)
#        except:
#            print("error for column:"+column+" in aggregatePerColumn")
#            pass
        g=df.groupby(by=column,as_index=False)
        if not parallel:
            g=g.agg(functions)
            g.columns = ['_'.join(col).strip()+"_over_"+column for col in g.columns.values]
            g[column]=g.index
        else:

            #there is a stupid issue with parallel apply and pandaparallel
            #in some cases, the aggregation function returns the aggregation column
            #in some other cases it does not.
            #Hence we are using this piece of code here with the try/catch block
            #in order to make sure that the final result also contains the original column
            #over which we aggregate
            #It is important that the first function is a function like the np.mean which leaves a column intact
            original_column=None
            res=[]
            for func in functions:
                result=g.parallel_apply(func)
                try:
                    if original_column is None:
                        original_column=result[column]
                    result.drop(column,inplace=True,axis=1)
                except:
                    pass
                result.columns=[col+'_'+func.__name__.strip()+"_over_"+column for col in result.columns]
                res.append(result)


            g=pd.concat(res,axis=1)
            g[column]=original_column
            g[column]=g[column].astype('category')
            
        
        groups.append(g)
        
    for g,column in zip(groups,columns):
        g=g.rename_axis(None)
        df=pd.merge(df,g,on=column,how="left")
    return df
    
def interactionNumerics(df):
    if np.all(df.dtypes=='category'):
        return df
    df=df.copy()
    columns=df.select_dtypes(include=[np.number]).columns

    res=[]
    colnames=[]
    #multiplication and division
    for col1 in columns:
        for col2 in columns:
            if col1!=col2 and col1.find('_over_')==-1 and col2.find('_over_')==-1:
                dummy1=df[col1]*df[col2]
                res.append(dummy1)
                colnames.append(col1+'_mult_'+col2)
                
                dummy2=df[col1]/df[col2]
                res.append(dummy2)
                colnames.append(col1+'_div_'+col2)

                
    g=pd.concat(res,axis=1)
    g.columns=colnames
    df=pd.concat([df,g],axis=1)
    
    return df
    


def createFeatures(df,aggregationfunctions=None,parallel_feature_creation=False):    
    if aggregationfunctions is None:
        aggregationfunctions = adan.functions.aggregationfunctions
    
    colsagg=aggregatePerColumn(aggregationfunctions,df,parallel=parallel_feature_creation)

    return colsagg
