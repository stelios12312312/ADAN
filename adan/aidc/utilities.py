# -*- coding: utf-8 -*-
import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)

from adan.aidc.feature_preprocess import *
from adan.aidc.feature_creation import *
import pandas as pd
from io import StringIO
from adan.functions import aggregationfunctions
from sklearn.ensemble import IsolationForest

#I think this function here is useless   
#def identity(x):
#    return x
def test_header(df,threshold=0.8):
    """
    Returns true if a percentage (threshold) of the columns are numbers.
    This a heuristic to identify that the columns have been misread, and the first
    column is not a header, but a row.
    """
    header=df.columns
    num_columns = len(df.columns)
    #if the header is like 0,1,2,3,4... then return
    if np.all(np.arange(num_columns)==header):
        return False
    
    
    #count how many columns seem to be floats
    floats=0
    for col in df.columns:
        #a column that can be converted to a date is a sign of this not being a header
        try:
            pd.to_datetime(col)
            return True
        except:
            pass
            
        try:
            float(col)
            floats+=1
        except:
            pass
        #there seems to eb some issue with ? as a character
        try:
            if col=='?':
                return True
        except:
            pass
        
    if floats/num_columns>=threshold:  
        return True
    return False


def detectOutliers(df):
    df2=df.select_dtypes('number')
    if df2.shape[1]<1:
        print('No numerical columns to calculate outliers on.')
        return None
    #don't use a crzy number of trees, to avoid taking up too much time
    n_estimators=min([df2.shape[0]*2,100])
    forest=IsolationForest(n_estimators=n_estimators,contamination=0.001)
    
    try:
        predictions=forest.fit_predict(df2)
        outliers=np.where(predictions==-1)[0]
    except:
        print('execution of outliers halted, maybe there are NaNs')
        return None
    
    return outliers
    
    
def readData(path,sep=None,header='infer',
             na_values=["?","n/a","na","NA","N/A"," ","NULL","NAN","nan","null","?","-","--","---"],type_of_file=None,
             encoding='utf-8',dtype=None,detect_outliers=True):
    """
    detect_outliers: If True, then this function will run an anomaly detection algorithm
    and add suggested outliers into the log, but will not remove them. This function 
    uses ONLY numerical attributes, as to avoid dimensionality explosion by mistake,
    since this requires the proper analysis. The entry log['outliers'] will return
    the row number of the outliers
    """
    
    log={}
    log['encoding']=encoding
    log['header']=header
    log['1000s_separator']=','
    
    if type(path) is type(pd.DataFrame()):
        return path,{}
    
    if type_of_file is None:
        if path.find('.csv')>-1:      
            type_of_file='csv'
        elif path.find('.xlsx')>-1 or path.find('.xls')>-1:
            type_of_file='excel'
        
    if type_of_file=='csv':
        
        if sep==None:
            sep=','
            
        #use different encoding if this fails
        try:
            #try with comma first
            df=pd.read_csv(path,sep=sep,na_values=na_values,header=header,encoding=encoding,thousands=',',dtype=dtype)
        except:
            encoding='latin1'
            df=pd.read_csv(path,sep=sep,na_values=na_values,header=header,encoding=encoding,thousands=',',dtype=dtype)

        
        if len(df.columns)<=1:
            f=open(path,'r')
            t=f.read()
            comma=t.count(',')
            semicolon=t.count(';')
            whitespace=t.count(' ')
            separators=[comma,semicolon,whitespace]
            if max(separators)==comma:
                sep=','
            if max(separators)==semicolon:
                sep=';'
            if max(separators)==whitespace:
                sep="\s+"
            #try with other seperators
            df=pd.read_csv(path,sep=sep,na_values=na_values,headers=header,encoding=encoding,thousands=',',dtype=dtype)
        log['separator']=sep
        
        #this is an additional test to infer the right header
        if test_header(df):
            df=pd.read_csv(path,sep=',',na_values=na_values,header=None,encoding=encoding,dtype=dtype)
            log['header']='no header'
        else:
            log['header']='header detected'
                
    elif type_of_file=="excel":
            df=pd.read_excel(path,na_values=na_values,encoding=encoding)
            if test_header(df):
                df=pd.read_excel(path,na_values=na_values,header=None,encoding=encoding,dtype=dtype)
                log['header']='no header'
            else:
                log['header']='header detected'
            
    else:
        return None
        
    
    #These are heuristics in order to understand if we should read the header
    #and if the first column contains indices or not
    num_floats_in_columns=0
    for col in df.columns:
        try:
            float(col)  
            num_floats_in_columns=num_floats_in_columns+1
        except:
            pass
        
    num_floats_in_first_row=0
    for item in df.iloc[0,:].values:
        try:
            float(item)  
            num_floats_in_first_row=num_floats_in_first_row+1
        except:
            pass
    
    if num_floats_in_columns==num_floats_in_first_row:
        df=pd.read_csv(path,sep=sep,na_values=na_values,header=None,encoding=encoding,dtype=dtype)
        log['header']='no header'
    else:
        log['header']='header detected'
    
    log['variable_types']=df.dtypes
    
    log['outliers']=None
    if detect_outliers:
        outs=detectOutliers(df)
        log['outliers']=outs
    
    
    return df,log
    
def createLags(dataframe,lagged_variables,lags=[1,2,3]):
    df=dataframe.copy()
    for var in lagged_variables:
        for lag in lags:
            col_name=var+'_lag'+str(lag)
            if col_name in df.columns:
                col_name=col_name+'adan_made'
            df[col_name]=df[var].shift(lag)
    
    df=df.dropna() 
    
    return df
    

def prepareTrainData(dataframe,task,target_name=None,del_zero_var=True,copy=True,fillNaN=True,
                     center=False,fix_skew=True,n_components=10,pca_criterion_thres=0.05,
                     parallel_feature_creation=False,createFeats=True):
    """
    center: avoid centering the values when this might cause issues with log 
    and square root
    """
    log={}
    filler={}    
    
    if copy:
        df=dataframe.copy()
    else:
        df=dataframe

           
    #drop the target variable, if the dataframe is to be used as input
    if target_name!=None:
        target=df[target_name].values
        target,target_category_mapper=convertTargetToNumerical(target)
        df,target,bad_target_values=preprocessTarget(df,target,task=task)
        df.drop([target_name],inplace=True,axis=1)
    else:
        target=[]
        target_category_mapper=None
        bad_target_values=None
    
    _,columns_removed = deleteIdColumn(df)
    log['removed_columns_because_detected_as_ID']=columns_removed
    
    #fixes column names
    df.columns=fixColumns(df.columns)    
    
    df.columns=["var_"+str(k) for k in df.columns]

    converted_column_dates=convertDates(df)
    log['date_columns']=converted_column_dates
    
    numerical_cols=try_to_make_numerical(df)
    categorical_vars=makeObjectToCategorical(df)
    
    zero_var_columns=[]
    if del_zero_var:
        zero_var_columns=deleteZeroVariance(df)
    log['zero_variance_columns_removed']=zero_var_columns
    


        
    if fillNaN:
        filler,columns_with_missing,to_remove=fillOutNaN(df)
        df.drop(to_remove,axis=1,inplace=True)
        
    categorical_vars=makeObjectToCategorical(df)
    log['categorical_variables']=categorical_vars.values
        
    log['columns_with_missing_values_that_were_filled'] = columns_with_missing
    log['columns_removed_due_to_missing_or_NaN']=to_remove

    if pca_criterion(df,threshold=pca_criterion_thres):
        print('PCA CRITERION TRIGGERED. RUNNING PCA WITH {0} COMPONENTS.'.format(n_components))
        df,components,pca_object=run_PCA_on_numerical(df,n_components=n_components)
    else:
        components=None
        pca_object=None

    if createFeats:
        df=createFeatures(df,parallel_feature_creation=parallel_feature_creation)
    else:
        pass
    # df=interactionNumerics(df)
    
    drop_cols=[]
    for col in df.columns:
        if df[col].unique().shape[0]==1:
            drop_cols.append(col)
    df.drop(drop_cols,axis=1,inplace=True)
    
    df=pd.get_dummies(df,prefix_sep='_categoryname_',columns=categorical_vars)    
    df=df._get_numeric_data()  
    
    df=df.astype(np.float32)
    
    if del_zero_var:
        deleteZeroVariance(df)
        
    #df.columns=fixColumns(df.columns)

        
    if fillNaN:
        filler2,_,to_remove=fillOutNaN(df)
        df.drop(to_remove,axis=1,inplace=True)
    log['columns_removed_due_to_missing_or_NaN']+=to_remove
    
    #merge the 2 fillers. The first filler was applied before the new features, the other filler
    #after the new features
    
    for key in filler2.keys():
        filler[key]=filler2[key]
    
    scaler={}
    centerer={}
    for column in df.columns:
        if column.find('_categoryname_')==-1:
            if df[column].var()>0:
                scaler[column]=df[column].var()
                if center:                    
                    centerer[column]=df[column].mean()
                    df[column]=(df[column]-centerer[column])/scaler[column]
                else:
                    df[column]=df[column]/scaler[column]

            #elif df[column].var()==0:
            #    df.drop(column,1)

    df.columns=fixColumns(df.columns)   
    #df.columns=pd.io.parsers.ParserBase({'names':df.columns})._maybe_dedup_names(df.columns)
    
    return df, target,scaler,centerer,categorical_vars,filler,log,\
target_category_mapper,numerical_cols,components,pca_object,bad_target_values


def prepareTestData(dataframe,scaler={},centerer={},model=None,categorical_vars_match=[],
                    match_columns=[],filler={},aggregationfunctions=aggregationfunctions,
                    numerical_variables=[],n_components=None,pca_object=None):
    
    df=dataframe.copy() 
    df=pd.DataFrame(df)      
    #NOTE, there is a rule here. If the dataset is less than 10 rows, then
    #we are not  calculating the IdColumn. The reason is that
    #it makes no sense to do this for small datasets. If you supply one row
    #for example, then it removes this row.
    deleteIdColumn(df)
    df.columns=fixColumns(df.columns)
    new_cols=[]
    for k in df.columns:
        if k.find('var_')==-1:
            new_cols.append("var_"+str(k))
        else:
            new_cols.append(str(k))
    df.columns=new_cols
    
    convertDates(df)
    
    
    if len(categorical_vars_match)>0:
        for cat in categorical_vars_match:
            df[cat]=df[cat].astype("category")
            
    for col in numerical_variables:
        if col in df.columns:
            df[col]=numeric_conversion(df[col])[0]           
    
    final_funs=[]
    for fun in aggregationfunctions:
        if str(model).find(fun.__name__)>-1:
            final_funs.append(fun)
            
        
    fillOutNaN(df,filler=filler)
        
    if pca_object is not None:
        df,_,_ = run_PCA_on_numerical(df=df,pca=pca_object)
        
        
    #remove columns not used in the model, in order to improve efficiency
    categorical_vars_match=np.array(categorical_vars_match)
    columns_to_remove = []
    for col in df.columns:
        if str(model).find(col)==-1:
            columns_to_remove.append(col)
            where = np.where(categorical_vars_match==col)[0]
            if len(where)>0:
                categorical_vars_match = np.delete(categorical_vars_match,where)
                
    df.drop(columns_to_remove,inplace=True,axis=1)
    
    df=createFeatures(df,final_funs)
    
    try:
        df=pd.get_dummies(df,prefix_sep='_categoryname_',columns=categorical_vars_match)    
    except:
        pass
    df=df._get_numeric_data()  
    
    df=df.astype(np.float32)

    #remove columns that do not exist in the original data, before creating features, excluding dummy columns
    if len(match_columns)>0:
        for col in df.columns:
                if col not in match_columns:
                    df.drop(col,axis=1,inplace=True)
    
        diff=set(match_columns)-set(df.columns)
        for col in diff:
            df[col]=0
                    
    fillOutNaN(df,filler=filler)
    
    for col,m in centerer.items():
        try:
            df[col]=df[col]-m
        except:
            pass
    
    for col,v in scaler.items():
        try:
            df[col]=df[col]/v
        except:
            pass
    df.columns=fixColumns(df.columns)
    

    return df
    
    
def sample(df,targets,fraction=0.5):
    n=df.shape[0]-1
    samples=np.random.permutation(n)
    samples=samples[0:int(np.round(fraction*n))]
    return df.ix[samples,:],targets[samples]