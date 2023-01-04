#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 09:54:14 2019

@author: stelios
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:20:53 2018

@author: stelios
"""

import pandas as pd
import scipy as sp
import scipy.stats
from scipy.stats import burr
import numpy as np
from helpers import *
import sys
import warnings

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    try:
        iters=int(sys.argv[1])
    except:
        iters=10
    try:
        num_rows=int(sys.argv[2])
    except:
        num_rows=100
    
    try:
        optim_iters=int(sys.argv[3])
    except:
        optim_iters=10
        
    path = sys.argv[4]
    output_file=sys.argv[5]
    if output_file is None:
        output_file='results_data.csv'
    
    try:
    #need to specify .csv of excel
        type_of_file=sys.argv[6]
    except:
        type_of_file=None
    if type_of_file is None:
        if path.find('.csv')>-1:
            print('Assumes the file is in .csv')
            type_of_file='csv'
        else:
            print('Assumes the file is in Excel')
            type_of_file='excel'
    
    df = readData(path,type_of_file=type_of_file)
    df=impute_missing(df)
    
    #main functions. Produces an artificial dataset (df_art) and some test metrics (error_total)
    df_art, error_total = core_function(df,num_rows=num_rows,iters=iters,
                                        optim_iters=optim_iters,use_vae=True)
    
    df_art.to_csv('results_file.csv',index=False)
    print("Total error is :"+str(error_total))