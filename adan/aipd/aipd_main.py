import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori,fpmax
import os

def associationRuleMining(df,num_partitions=3,support=0.33,columns_limit=2000,max_len=4):
    """
    Performs association rule mining. It partitions numerical features into a number
    of partitions based on quantiles.
    
    columns_limit: if the final number of columns is > limit, then return with an error
    """
    if np.all(df.dtypes=='category'):
        print('Warning: The variables are all categorical.')
        return None
    
    if df.shape[1]>columns_limit:
        print('Error: The number of columns is larger than the allowable limit. The algorithm is likely to require a very long time to converge.')
        return None
    
    df_new=pd.DataFrame()    
        
    for col in df.columns:
        if df[col].dtype!='object':
            df_new[col]=pd.cut(df[col],num_partitions,duplicates='drop')
        else:
            df_new[col]=df[col].copy()
            
    df_new=pd.get_dummies(df_new)
    df_new=df_new.astype(bool)
        
    res=fpmax(df_new, min_support=support, use_colnames=True,max_len=max_len)
    res['length'] = res['itemsets'].apply(lambda x: len(x))
    res=res[res['length']>=2]
    #filter out rules, to keep only the most powerful ones
    res=res[res.support>=res.support.median()]
    
    return res

def detectHighCorrelations(df,threshold=0.3,columns_limit=1000):
    if np.all(df.dtypes=='category') or np.all(df.dtypes=='object'):
        print('Warning: The variables are all categorical.')
        return None
    
    if df.shape[1]>columns_limit:
        print('Error: The number of columns is larger than the allowable limit. The algorithm is likely to require a very long time to converge.')
        return None
    corr_matrix = df.corr()

    #the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
    sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                     .stack()
                     .sort_values(ascending=False))
    
    sol_filtered=sol[abs(sol)>threshold]
    
    return sol_filtered
    

def runAiPD(df,high_cor_threshold):
    high_cors=findHighCorrelations(df,high_cor_threshold)
    association_results=associationRuleMining(df)
    aiem_results=smartAiEM(df,population=100,generations=50)
    
    

# dataset_path = os.path.join(
#     os.environ['DATASETS_DIR'], 'heart_disease', 'heart_disease.csv')

# df = pd.read_csv(dataset_path, header=None)#res=associationRuleMining(df)
# sol=detectHighCorrelations(df)
