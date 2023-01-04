# -*- coding: utf-8 -*-
import numpy as np
from dateutil.parser import parse
import pandas as pd
import warnings
import scipy as sp
from sklearn.decomposition import IncrementalPCA
from pandas.api.types import is_string_dtype


def fixColumns(columns):
    """
    fixColumns fixes problems that might exist with the names of the columns in the dataframe.
    """
    cols=[]
    for col in columns:
        if type(col)!=str:
            col="var"+str(col)
        col=col.strip().replace(',','__comma__')
        col=col.strip().replace(" ","_").replace(".","_dot_").replace("-","_").replace("@","at").replace("(","").replace(")","").replace("%","_percent_")
        col=col.strip().replace("+","_plus_").replace("/","_div_").replace("*","_star_").replace("\'","")        
        cols.append(col)
    return cols
    
def convertTargetToNumerical(x):
    mapper={}
    #there had been an incidence with a dataset, where one category was mistakenly 'A '
    try:
        x=np.array([k.strip() for k in x])
    except:
        pass
    if isinstance(x[0],str):
        elements=np.unique(x)
        for i in range(0,len(elements)):
            x[x==elements[i]]=i
            mapper[elements[i]]=i
        x=x.astype('int32')
    return x,mapper

def deleteZeroVariance(df,unique_threshold=0.95):
    """
    unique_threshold: If the percentage of unique values is above this threshold
    then the column has so low variance that we have to assume it is like an id column and 
    remove it
    """
    
    toremove=[]
    for column in df.columns:
        if df[column].dtypes.name=="category":      
            if len(df[column].cat.categories)==1:
                toremove.append(column)
            if df[column].unique().shape[0]/df.shape[0]>unique_threshold:
                toremove.append(column)
   
        #if all the values are missing, then drop the column
        elif df[column].isnull().sum()==df.shape[0]:
            toremove.append(column)
        else:
            if not df[column].std()>0.0:
                toremove.append(column)   


    df.drop(toremove,axis=1,inplace=True)
    return toremove
    
def fillOutNaN(df,numerical_filler=np.nanmedian,filler={},upper_bound_missing=0.4):
    """
    replaces NA values while also return a 'filler' dictionary with the value 
    that should replace a missing value
    in the test set.
    
    If there is more % than the upper_bound_missing_count, then this variable shouldbe removed instead.
    Removal is handled outside the function.
    
    There is no explcit to specify whether the function is used in Training or Test
    other than providing a filler object, which contains the values that should replace the 
    missing values of each column. If a missing value is observed at test time, but filler is empty for this column, 
    then it is imputed.
    
    """
    
    dummy={}
    columns_with_missing=[]
    to_remove=[]
    for col in df.columns:
#        try:
        missing_count=df[col].isnull().sum()/df.shape[0]
        if missing_count>0 and missing_count<upper_bound_missing:
            if df[col].dtype.name=="category":
                #if the column does not exist in the filler, then impute
                if len(filler)>0 and (col in filler.keys()):
                    fill=filler[col]
                else:
                    fill=df[col].mode()[0]
                df[col].fillna(value=fill,inplace=True)
            else:
                df[col].replace([np.inf, -np.inf], np.nan,inplace=True)
                #if a filler is provided, 
                if len(filler)>0 and (col in filler.keys()):
                    fill=filler[col]                        
                else:
                    fill=numerical_filler(df[col])
                df[col].fillna(value=fill,inplace=True)
            dummy[col]=fill
            columns_with_missing.append((col,missing_count))
                
        elif missing_count>=upper_bound_missing:
            print('excessive number of missing values for column: '+str(col))
            to_remove.append(col)
#        except:
#            print('could not fill out missing values for '+col)
        
    return dummy,columns_with_missing,to_remove

def numeric_conversion(column):
    final=[]
    for element in column.values:
        if type(element)==type(''):
            element = element.replace(',','')
        try:
            final.append(float(element))
        except:
            return column.values,False
    return final,True

def try_to_make_numerical(dataframe):
    columns=dataframe.columns
    numerics=[]
    for col in columns:

        dataframe[col],is_numeric=numeric_conversion(dataframe[col])
        if is_numeric:
            numerics.append(col)  
    return numerics          

def makeObjectToCategorical(df):
    categorical_vars=[]
    columns=df.select_dtypes(include=[object]).columns
    for column in columns:
        df[column]=df[column].str.strip()
        df[column]=df[column].astype('category')
    
    categorical_vars = df.select_dtypes(include=['category']).columns

    return categorical_vars
    
      
      
def checkIfDate(column):
    """
    Functions that uses heuristics to determine if a column is a date column.
    """
    num_dates=0
    for element in column:
        try:
            #an element must contain at least 6 characters to be a date, e.g. 010116
            if(len(element)>=6):
                parse(element)
                num_dates=num_dates+1
        except:
            pass
    if num_dates>(len(column)/2):
        return True
    else:
        return False
    
    
def convertDates(df,day=True,month=True,year=True,hour=True):
    columns=df.select_dtypes(include=[object]).columns
    converted_columns_dates=[]
    for column in columns:
        if checkIfDate(df[column]):
            try:
                date=pd.to_datetime(df[column])
                if(day):
                    df['day']=[x.day for x in date]
                    df['day']=df['day'].astype('category')
                if(month):
                    df['month']=[x.month for x in date]
                    df['month']=df['month'].astype('category')
                if(year):
                    df['year']=[x.year for x in date]
                    df['year']=df['year'].astype('category')
                if(hour):
                    df['hour']=[x.hour for x in date]
                    df['hour']=df['hour'].astype('category')
                df.drop(column,axis=1,inplace=True)
                converted_columns_dates.append(column)
            except:
                pass
    return converted_columns_dates
    


def deleteIdColumn(df):    
    columns_removed=[]
    if df.shape[0]<10 or type(df)==pd.core.series.Series:
        return df,columns_removed
    for col in df.columns:
        try:
            if df[col].unique().shape[0]==df.shape[0] and is_string_dtype(df[col]):
                df.drop(col,axis=1,inplace=True)
            elif col.lower().strip()=='id':
                df.drop(col,axis=1,inplace=True)
                columns_removed.append(col)
            elif col.lower().find('unnamed:')>-1:
                df.drop(col,axis=1,inplace=True)
                columns_removed.append(col)
        except:
            pass
    return df,columns_removed


def generalPreprocess(newd,add_var_name=True,del_zero_var=True):
    convertDates(newd)
    categorical_vars=makeObjectToCategorical(newd)
    fillOutNaN(newd)
    if del_zero_var:
        deleteZeroVariance(newd)
    deleteIdColumn(newd)
    if add_var_name:
        newd.columns=["var_"+str(k) for k in newd.columns]    
    return newd,categorical_vars
    
def preprocessTarget(df,target,task='classification',fix_skew=True):
    #delete rows with lots of missing values
    bad_target_values=[]
    if not np.isfinite(target).all():
        print('SOME TARGET VALUES ARE NaN. REMOVING THEM FROM DATASET.')
        #Indices of bad target values which are either inf or nan
        bad_target_values=np.where(np.logical_not(np.isfinite(target)))[0]
    df2=df[np.isfinite(target)]
    target2=target[np.isfinite(target)]
    rows_original=df.shape[0]
    rows_new=df2.shape[0]
    if rows_new<=rows_original/2.0:
        warnings.warn("half of the rows were removed due to missing values in the target variable!")
        
    if task=='classification':
        #make sure that in binary classification problems, the classes are always 0 and 1
        #multiclass problems become 0,1,2 etc.
        unique_values=np.unique(target2)
        for counter,un in enumerate(unique_values):
            #the reason we subtract 11 is because we want to avoid a situation, where in a problem
            #with classes [-1,0,1] class -1 becomes 0
            #and then class 0 and class -1 (now turned 0) both become 1, which messes things up
            target2[target2==un]=counter-11
        
        target2=target2+11
    
    
    return df2,target2,bad_target_values
    
def pca_criterion(df,threshold=0.05):
    
    """
    Determines whether it is a good idea to run PCA.
    """
    cors = np.abs(df.corr())
    mean = cors.mean()
    std = cors.std()
    
    criterion=np.mean(mean-std)
    print('Value of PCA criterion is '+str(criterion))
    if np.mean(mean-std)>threshold:
        return True
    
    return False

def run_PCA_on_numerical(df,n_components=10,pca=None):
    #create a new DF to silence the warning
    numbers=pd.DataFrame(df.select_dtypes(np.number))
    categories=pd.DataFrame(df.select_dtypes('category'))
    components=None
    
    if n_components>numbers.shape[1]:
       n_components=numbers.shape[1] 
    
    if pca is None:
        pca=IncrementalPCA(n_components=n_components)
        
        means=numbers.mean()
        stds=numbers.std()
        
        numbers=(numbers-means)/stds
        
        dfpca=pca.fit_transform(numbers)
        pca.columns=numbers.columns
        pca.means=means
        pca.stds=stds
        
        indexname=[]
        for col in range(pca.components_.shape[0]):
            indexname.append('var_PCA_'+str(col+1))
        components=pd.DataFrame(pca.components_,columns=numbers.columns,index=indexname)
    else:
        numbers=numbers[pca.columns]
        
        numbers=(numbers-pca.means)/pca.stds
        
        dfpca=pca.transform(numbers)
    
    for col in range(dfpca.shape[1]):
        colname='var_PCA_'+str(col+1)
        categories[colname]=dfpca[:,col]
               
    return categories,components,pca
    
    
    

    
