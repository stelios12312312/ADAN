#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:55:04 2018

@author: stelios
"""
import pandas as pd
import scipy as sp
import scipy.stats
from scipy.stats import norm,multivariate_normal
import numpy as np
import scipy
import scipy.stats

from datetime import datetime  
from datetime import timedelta


def readData(path,sep=None,header='infer',na_values=["?","na","NA","N/A"," ","NULL","NAN","nan","null"],type_of_file='csv'):
    """
    This is a copy of the readData function from the AIDC module of ADAN.
    It is copied here to make the synthetic data module autonomous.
    """
    if type_of_file=='csv':
        
        if sep==None:
            #try with comma first
            df=pd.read_csv(path,sep=',',na_values=na_values,header=header)
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
            df=pd.read_csv(path,sep=sep,na_values=na_values,header=header)
    elif type_of_file=="excel":
            df=pd.read_excel(path,na_values=na_values)
    else:
        print('Warning: Type of file not specified. Returning none.')
        return None
    
        
    if type(path) is type(pd.DataFrame()):
        return path
        
    
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
        df=pd.read_csv(path,sep=sep,na_values=na_values,header=None)
    
    return df

#def calc_error_vector(df_art,df_original):
#    """
#    The same function like below, but for a single column. Only difference
#    is the absence of normalisation
#    """
#
#    original_means=df_original.mean()
#    original_std=df_original.std()
#    original_skew=scipy.stats.skew(df_original)
#    original_kurtosis=scipy.stats.kurtosis(df_original)
#    original_cov = scipy.stats.variation(df_original)
#    original_mode = scipy.stats.mode(df_original)[0][0]
#    
#    art_means=df_art.mean()
#    art_std=df_art.std()
#    art_skew=scipy.stats.skew(df_art)
#    art_kurtosis=scipy.stats.kurtosis(df_art)
#    art_cov=scipy.stats.variation(df_art)
#    art_mode=scipy.stats.mode(df_art)[0][0]
#    
#    score = np.mean(abs(original_means - art_means)+abs(original_std - art_std)+
#                    abs(original_skew - art_skew)+
#                    abs(original_kurtosis - art_kurtosis)+
#                    abs(original_cov-art_cov))
#    
#    return score


def calc_error(df_art,df_original,normalization=True):
    """
    This function is used in order to calculate the error of the fitting distribution.
    It simply calculates a bunch of statistics and compares the statistics produced by the actual 
    fit and the real data. The final error metric is based on the mean absolute error between
    the two statistics.
    """
    original_means=np.std(df_original)
    original_std=np.std(df_original)
    original_skew=scipy.stats.skew(df_original)
    original_kurtosis=scipy.stats.kurtosis(df_original)
    original_cov = scipy.stats.variation(df_original)
    original_mode = scipy.stats.mode(df_original)[0][0]
    
    art_means=np.mean(df_art)
    art_std=np.std(df_art)
    art_skew=scipy.stats.skew(df_art)
    art_kurtosis=scipy.stats.kurtosis(df_art)
    art_cov=scipy.stats.variation(df_art)
    art_mode=scipy.stats.mode(df_art)[0][0]
    
    #Need to normalize when comparing dataframes other one statistic might end up dominating
    #normalisation is not needed for columns
    if normalization:
        original_means, art_means = normalize(original_means,art_means)
        original_std, art_std = normalize(original_std,art_std)
        original_skew, art_skew = normalize(original_skew,art_skew)
        original_kurtosis, art_kurtosis = normalize(original_kurtosis,art_kurtosis)
        original_cov, art_cov = normalize(original_cov,art_cov)
        original_mode, art_mode = normalize(original_mode,art_mode)
    
    #the first part is attempted in case we have dataframes. In that case we 
    #take the mean score of the sum across all columns. Otherwise return a single number.
    try:
        score = np.mean(sum(abs(original_means - art_means))+sum(abs(original_std - art_std))+
                        sum(abs(original_skew - art_skew))+
                        sum(abs(original_kurtosis - art_kurtosis))+
                        sum(abs(original_cov-art_cov)))
    except:
        score = np.mean(abs(original_means - art_means)+abs(original_std - art_std)+\
                abs(original_skew - art_skew)+\
                abs(original_kurtosis - art_kurtosis)+\
                abs(original_cov-art_cov))
    
    return score

def find_distribution(column):
    """
    Tries to fit all distributions in scipy on a single column of data.
    Chooses a winner based on either the error or AIC.
    
    Returns a tuple:
        final_error: winning distribution's name and parameters based on the error metric
        aic_error: winning distribution's name and parameters based on AIC
    """
    y = column.values
        
    dist_names=dir(scipy.stats)
    dist_names.remove('gausshyper')
    dist_names.remove('argus')
    dist_names.remove('levy_l')
    dist_names.remove('halflogistic')
    dist_names.remove('gumbel_l')
    dist_names.remove('levy_stable')
    dist_names.remove('rdist')
    #dist_names.remove('ncf')
    #dist_names=['levy_stable']

    loglik={}
    dist_params={}
    p_value = {}
    data = {}
    params = {}
    AIC = {}
    for dist_name in dist_names:
        try:
            #print('trying '+str(dist_name))
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y,maxiter=10)
            params[dist_name] = param
            logl = dist.logpdf(y,*param[:-2],loc=param[-2],scale=param[-1]).sum()
            loglik[dist_name] = logl
            dist_params[dist_name] = param
            pvalue = scipy.stats.kstest(y,dist.cdf,args=param).pvalue
            #AIC type 2: http://www.brianomeara.info/tutorials/aic/
            aic = 2*(len(param))*(len(y)/(len(y)-len(param)-1)) - 2*np.log(logl)
            p_value[dist_name] = p_value
            y_sorted=np.sort(y)
            rands = np.sort(dist.rvs(size=len(y),*param))
            #the error (which is the best metric) is calculated by a function
            #that incorporate some heuristics. Check calc_error() above
            error = calc_error(y_sorted,rands,normalization=False)    
            AIC[dist_name] = aic
            data[dist_name] = (logl,pvalue,aic,error)
        except:
            pass
        
    res = pd.DataFrame.from_dict(data,orient='index',
                                 columns=['loglik','pvalue','aic','error'])
    res=res.sort_values(by='error',ascending=True)
    res=res.replace([np.inf, -np.inf], np.nan)
    
    res2=res.dropna(subset=['error','aic'])
    #in some cases we get an empty dataframe, because AIC cannot be calculated
    #if that's the case then revert to the previous one
    if res2.shape[0]==0:
        res2=res
    #positive loglikelihood indicates a bug
    res2=res[res['aic']<0]
    if res2.shape[0]>0:
        res=res2
    #res=res[res['pvalue']>0]
#    res['super_strong'] = res.pvalue<0.001
#    res['strong'] = res.pvalue<0.01
#    res['moderate'] = res.pvalue<0.05
#    res['weak'] = res.pvalue<0.1
#    
#    if(sum(res['super_strong'])>0):
#        res = res[res['super_strong']]
#    elif (sum(res['strong'])>0):
#        res = res[res['strong']]
#    elif (sum(res['moderate'])>0):
#        res = res[res['moderate']]
#    elif (sum(res['weak'])>0):
#        res = res[res['weak']]

    #get winnner by error
    winner_name_error = res.iloc[0,:].name

    final_error = (winner_name_error, params[winner_name_error])
    
    #get winnner by AIC.
    #Sometimes AIC can't be calculated due to likelihood issues, fall back
    #to error
    try:
        res=res.sort_values(by='aic',ascending=True)
        winner_name_aic = res.iloc[0,:].name
        final_aic = (winner_name_aic, params[winner_name_aic])
    except:
        final_aic=final_error.copy()

    return final_error, final_aic
    
    
def identify_dists(df):
    """
    Find the best fit distribution for each column of a dataframe.
    """
    dists_error = {}
    dists_aic = {}
    for i in np.arange(df.shape[1]):
        print('Processing column ' +str(i+1) +' out of '+str(df.shape[1]))
        result_error, result_aic = find_distribution(df.iloc[:,i])
        dists_error[df.columns[i]] = result_error
        dists_aic[df.columns[i]] = result_aic
        
    return dists_error, dists_aic

def run_copula(dists,df):
    """
    Fits the copula model
    """
    new_df = df.copy()
    for i in range(0,df.shape[0]):
        row = new_df.iloc[i,:]
        new_row=[]
        for j in range(len(row)):
            col_name = row.index[j]
            dist_tuple = dists[col_name]
            dist = getattr(scipy.stats,dist_tuple[0])
            cdf_result = dist.cdf(row[j],*dist_tuple[1])
            qnorm_res = norm.ppf(cdf_result)
            
            new_row.append(qnorm_res)
        new_df.iloc[0,:]=new_row      
    df_copula=new_df.replace([np.inf, -np.inf], np.nan)
    df_copula.dropna(inplace=True)
    return df_copula    

def generate_data(df_copula,dists,df_original=None,num_rows=100):
    """
    Generate random variables, feed them into the copula and then do the inverse
    process in order to convert them from the Uniform distribution to the
    actual distribution the data is following.
    """
    
    #art_data = multivariate_normal.rvs(mean=df_copula.mean(),
    #                                   cov=df_copula.corr(),size=num_rows)
    
    art_data = np.random.randn(num_rows,df_copula.shape[1])
    A=np.linalg.cholesky(df_copula.corr())
    
    for a in range(len(art_data)):
        art_data[a,:]=A.dot(art_data[a,:])
       
    art_data=pd.DataFrame(art_data,columns=df_copula.columns)

    new_rows = []
    for i in range(len(art_data)):
        row = art_data.iloc[i,:]
        new_row=[]
        for j in range(len(row)):
            col_min = df_original.iloc[:,j].min()
            col_max = df_original.iloc[:,j].max()
            col_name = row.index[j]
            dist_tuple = dists[col_name]
            dist = getattr(scipy.stats,dist_tuple[0])
            cdf_norm = norm.cdf(row[j])
            if cdf_norm<1 and cdf_norm>0.001:                
                quantile_result = dist.ppf(cdf_norm,*dist_tuple[1])
            #if the value of the CDF is too low or too high simply replace
            #with the minimum value or the maximum value found int he column
            #Otherwise we risk the possibility of getting weird values, such as NA, or Inf
            elif cdf_norm<0.001:
                quantile_result = col_min
            else:
                quantile_result = col_max
                
            if quantile_result>col_max:
                quantile_result=col_max
                
            if quantile_result<col_min:
                quantile_result=col_min
                
            new_row.append(quantile_result)
            
        new_rows.append(new_row)
    final = pd.DataFrame(new_rows,columns=art_data.columns)
    return final

#add min and max functionality
def generate_find_best(df,dists,num_rows=100,iters=10):
    """
    Repeats the data generation process many times and returns the best dataset found
    as measured by the error score.
    """
    datasets = []
    scores = []
    df_copula = run_copula(dists,df)
    for i in range(0,int(iters)):
        print('generating data iteration '+str(i))
        new_dat = generate_data(df_copula,dists,df,int(num_rows*1.1))
        
        new_dat=new_dat.replace([np.inf, -np.inf], np.nan)
        new_dat.dropna(inplace=True)
        
        if(new_dat.shape[0]>num_rows):
            new_dat=new_dat.iloc[0:num_rows,]
        
        datasets.append(new_dat)
        score = calc_error(df,new_dat)
        scores.append(score)
    scores=np.array(scores)
    success = np.where(scores==scores.min())
    return datasets[success[0][0]], datasets


def normalize(data1,data2):
    combined = np.concatenate((data1,data2))
    mean = combined.mean()
    std = combined.std()
    
    return (data1-mean)/std,(data2-mean)/std
    
    
def optimise(art_df,supporting_df,df_original,iters=1000):
    """
    Adds and removes rows of data in a greedy-optimisation style in order
    to find a new dataset with a lower error.
    """
    supporting_df = supporting_df.copy()
    #supporting_df = pd.concat(supporting_df)
    supporting_df.reset_index(inplace=True,drop=True)
    
    art_df = art_df.copy()
    score_initial = calc_error(art_df,df_original)
    scores=[]
    for i in range(iters):
        print(i)
        index1=np.random.randint(0,art_df.shape[0])
        index2=np.random.randint(0,supporting_df.shape[0])
        
        row1 = art_df.iloc[index1,:].copy()
        row2 = supporting_df.iloc[index2,:].copy()
        
        score_previous = calc_error(art_df,df_original)
        #make the switch of the rows, calculate the new error
        art_df.iloc[index1,:] = row2.copy()
        score_now = calc_error(art_df,df_original)
        
        #if error is larger then abort the switch
        if score_now>score_previous:
            print('switching back')
            art_df.iloc[index1,:] = row1.copy()
            print(score_now)
            print(score_previous)
        else:
            supporting_df.drop(index2,inplace=True)
            supporting_df.reset_index(inplace=True,drop=True)
            
        scores.append(calc_error(art_df,df_original))
    
    score_now = calc_error(art_df,df_original)
    
    improvement = score_now - score_initial
    return art_df,improvement
        
def process_dataset(df):
    """
    This function is used to go through the dataset and discover if there
    are any columns that are categorical or dates and need special treatment.
    """
    new_df=df.copy()
    params = {}
    date_params = {}
    for i in range(df.shape[1]):
        try:
            if np.issubdtype(df.iloc[:,i],np.datetime64) or df.iloc[:,i].dtype=='object':
                df.iloc[:,i]=pd.to_datetime(df.iloc[:,i])
        except:
            pass
        if df.iloc[:,i].dtype=='bool' or df.iloc[:,i].dtype=='object':
            new_column, thresholds = process_categorical(df.iloc[:,i])
            new_df.iloc[:,i]=np.array(new_column).copy()
            params[df.columns[i]]=thresholds
        elif df.iloc[:,i].dtype=='datetime64[ns]':
            new_column, date_p = process_dates(df.iloc[:,i])
            new_df.iloc[:,i]=np.array(new_column).copy()
            date_params[df.columns[i]]=date_p
            
    return new_df, params, date_params
            
def reverse_categoricals(df,params):
    """
    Categorical variables require special treatment (please read paper for details)
    This function takes the parameters can converts the single number into categories
    """
    new_df = df.copy()
    for k in params.keys():
        new_column = []
        p=params[k].copy()
        column = df.loc[:,k].values
        for i in range(len(column)):
            #less than a occurs only when normalization got screwed
            p['larger_than_a']=np.logical_or(column[i]>=p.a,column[i]<p.a)
            #larger than b occurs only when normalization got screwed
            p['smaller_than_b']=np.logical_or(column[i]<=p.b,column[i]>p.b)
            p['between']=np.logical_and(p['larger_than_a'],p['smaller_than_b'])
            value = p[p['between']].index[0]
            new_column.append(value)
        new_df.loc[:,k] = new_column
    return new_df
        
    
def reverse_dates(df,params):
    new_df = df.copy()
    for k in params.keys():
        resolution = params[k]['resolution']
        start_date = params[k]['start_date']
        new_column = []
        p=params[k]
        column=df.loc[:,k].values
        for i in range(len(column)):
            value = int(column[i])
            if resolution=='hours':
                value = start_date + timedelta(hours=value)
            elif resolution=='days':
                value = start_date + timedelta(days=value)
            else:
                value = start_date + timedelta(seconds=value)

                
            new_column.append(value)
        new_df.loc[:,k] = new_column.copy()
    return new_df

        
def process_categorical(column):
    """
    Categoricals are encoded as a single number (read paper for details)
    """
    proportions=column.value_counts()/len(column)
    proportions=proportions.sort_values(ascending=False)
    thresh_b=[proportions[0]]
    for i in range(1,len(proportions)):
        thresh_b.append(thresh_b[i-1]+proportions[i])
        
    thresh_a = [0]+thresh_b[0:len(thresh_b)-1]
    
    thresholds=pd.DataFrame({'a':thresh_a,'b':thresh_b},index=proportions.index)    
    
    new_column=[]
    
    for i in range(len(column)):
        value = column.values[i]
        thresh = thresholds.loc[value,:]
        mu = (thresh.b - thresh.a)/2+thresh.a
        sigma = (thresh.b - thresh.a)/6
        r = np.random.randn()*sigma+mu
        new_column.append(r)
    
    return new_column, thresholds

def process_dates(column,resolution='hours'):
    """
    Convert dates to hours/days/seconds or smt else
    since some start date, where start date is taken by default to be the minimum 
    value (that is the earliest date).
    """
    start_date = column.min()
    new_col = column.copy()
    new_col = new_col-start_date
    
    #defaults to seconds
    if resolution=='days':
        new_col=new_col.dt.total_seconds()/(60*60*24)
    elif resolution=='hours':
        new_col = new_col.dt.total_seconds()/(60*60)
    else:
        new_col = new_col.dt.total_seconds()
        
    params = {'start_date':start_date,'resolution':resolution}
    return new_col, params
    

def median_or_mode(column):
    try:
        return column.median()
    except:
        return column.mode()
    
def impute_missing(df):
    
    return df.apply(lambda x: x.fillna(median_or_mode(x)),axis=0)

def core_function(df,num_rows=None,iters=10,optim_iters=100,use_vae=False):
    """
    This is the main function for generating data.
    
    It runs two processes, one optimises the error, the other optimises AIC.
    
    num_rows: how many rows to return. If none then num_rows=rows_original
    iters: the number of iterations for each optimisiaton method
    optim_iters: the number of iters for the last stage optimisation greedy algorithm,
    that replaces rows between datasets.
    
    """
    
    if num_rows is None:
        num_rows=df.shape[0]
        
    #df_original = df.copy()
    df,params,date_params = process_dataset(df)
    res_error, res_aic = identify_dists(df)
    
    #df_copula_error = run_copula(res_error,df)
    #df_copula_aic = run_copula(res_aic,df)
    
    print('run copula and generate data')
    data1, datasets_error = generate_find_best(df,res_error,num_rows=num_rows,iters=iters)
    data2, datasets_aic = generate_find_best(df,res_aic,num_rows=num_rows,iters=iters)
    
    datasets_error = pd.concat(datasets_error)
    datasets_aic = pd.concat(datasets_aic)
    
    datasets = pd.concat([datasets_error,datasets_aic])
    datasets.reset_index(drop=True)
    
    d1,error1 = optimise(data1,datasets,df,iters=optim_iters)
    d2,error2 = optimise(data2,datasets,df,iters=optim_iters)

    d_final,error_final = optimise(d1,d2,df,iters=int(optim_iters/2))
    if use_vae:
        #avoid cyclic imports
        from vae_model import run_vae 
        df_vae, vae_error=run_vae(num_rows=num_rows*10,df=df)
        d_final,error_final = optimise(d_final,df_vae,df,iters=int(optim_iters/2))
    
    error_total = calc_error(d_final,df)
    
    df_art=reverse_categoricals(d_final,params)
    df_art=reverse_dates(df_art,date_params)
    return df_art, error_total
