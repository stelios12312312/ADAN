# Protocols.py

Protocols.py exposes all core ADAN functionality in terms of predefined data analysis protocols. ADAN contains many bits and pieces. Each protocol consists of a certain workflow.

The caveat with the protocols is that each protocol contains many steps/algorithm, and each step is defined by many different parameters. 



# run_all()

This is the main endpoint for communication with ADAN. 

    def run_all(path_or_data,target_name,task='infer',quant_or_num=0.6,ngen=20,sample_n=-1,max_tree=3,
                 n_pop=[50,50,50,50],n_features=3,store_results='results_tests.csv',allowed_time=6,
                 test_perc=0.15,choice_method='quant',fix_skew=True,limit_n_for_feature_selection=1000,
                 target_sampling=0.9,pca_criterion_thres=0.05,selection_variable_ratio=0.1,
                 extract_patterns=False,causal=False,ngen_for_second_round=10,npop_for_second_round=10,
                 crossover_second_round=0.5,
                 mut_prob_second_round=0.1,individual_mut_second_round=0.1,fix_margins_low=False,
                                        fix_margins_high=False):
        
    path_or_data:Either a string path to a .csv/Excel or a pandas dataframe
    task: this can be either 'regression','classification' or 'infer' (which infers the task automatically)
        
    choice_method: The method of choice for choosing thebest features. Choices are 'num' 
    and 'quant'. Num means that only the best X features will be chosen. Quant means that 
    features with score>quantile will be chosen. E.g. choosing quant_or_num=0.5 will keep
    features with above median score.
    
    quant_or_num: related to choice method
    
    ngen: The number of generations for the genetic algorithm
    
    max_tree: The maximum depth of the tree used by the genetic programming algorithm
    
    n_pop: The population. This taks a list because we are using the islands version of genetic algorithms
    
    n_features: The final number of features to ue for the equation
    
    store_results: Where to save the results
    
    allowed_time: The allowed execution time of the genetic algorithm in minutes. The actual
    running time might be slightly longer due to some additional processing that need to be done.
    
    test_perc: ADAN can perform a train/test split. The determines the % for the test split.
    
    fix_skew: If True, then ADAN will test to see whether the data is skewed. If yes
    
    sample_n: If -1 then all the rows of the dataset are used. Otherwise use only this number of samples.
    limit_n_for_feature_selection: How many rows to use when selecting features. We need to take 
    a sample, otherwise it takes a very long time. This is a very important parameter. Ideally we want
    to use as a big of a sample as possible.
    
    pca_criterion_threshold: Calculate correlations between features. Get the mean and the std.
    If mean-std>threshold, then run PCA.
    
    target_sampling: When the features are evaluated, we can sample a % of the targets, and evaluate
    the performace on this subset. This should help with overfitting and finding better solutions.
    
    selection_variable_threshold: This is defined as (number of variables)/(number of rows). If this
    ratio is above the threshold, then perform feature selection. Feature selection can be very expensive, 
    so, it is not worth doing when there are many more rows than columns.
    
    causal: Whether to run causal analysis or not
    
    extract_patterns: Whether to extract patterns or not. This is done through the use of frequent item
    mining algorithms, correlations and other heuristics.
    
    ***these variables are used for the genetic algorithm in run_Equatin_model step 2***
    ngen_for_second_round: *
    npop_for_second_round: *
    crossover_second_round: *
    mut_prob_second_round: *
    individual_mut_second_round: if a mutation has been decided to occur, 
    then each bit of the feature vector [0,1,1,1,0,...,1] will be mutated according to this probability
    
    fix_margins_low/high: This variable refers to whether the predictions should go
    below or above the minimum/maximum target value in original dataset (e.g. some values
    can't be below 0). It makes sense to turn it to True in some contexts, but this requires domain
    knowledge. In general, setting it to True is a safe choice, unless we care about extrapolation.

Returns:

A dictionary with the following values. Some of these are self-explanatory, some of these relate to diagnostic output.

	'model':results_em['model'],
	'other_models':results_em['final_models'],
	'most_important_variables':variable_importance_results['most_important_variables'],
	'variable_importance_natural_language':variable_importance_results['log'],
	'general_patterns_rules':pattern_extraction['rules'],
	'general_patterns_correlations':pattern_extraction['correlations'],
	'causal_results_natural_language':causal_results['causal_results_interpretation']   ,
	'model_interpretation_natural_language':results_em['natural_language_interpretation'],
	'performance_test':results_em['performance_test'],
	'log_transform_applied_to_target':log_transform_applied,
	'performance_train':results_em['performance_train'],
	'shapley_summary':variable_importance_results['shapley_summary'],
	'processed_input_data':results_em['processed_input_data'],
	'variable_importance_results_dict':variable_importance_results,
	'results_em_dict':results_em,
	'successful_execution':results_em['successful_execution'],
	'predicted_train_values':results_em['predicted_train_values'],
	'predicted_test_values':results_em['predicted_test_values'],
	'pca_components':results_em['components'],
	'ground_truth_train':results_em['ground_truth_train'],
	'ground_truth_test':results_em['ground_truth_test'],
	'issues_log':results_em['issues_log'],
	'natural_language_PCA':variable_importance_results['components_explanation'],
	'realizer':results_em['sentence_realizer'],
	'constraints':constraints
How to use this dictionary. You can find below the objects which are relevant for the frontend.

**model**: The 'model' object contains the model. Read below for the documentation regarding this object.

**'most_important_variables'**: List of the variables deemed most important for the model by ADAN.

'**variable_importance_natural_language**': String. Natural langauge explanation of the most important variables

'**general_patterns_rules**': Part of the pattern extraction insights module.Creates a table of frequent patterns 

'**general_patterns_correlations**': Table of the highest correlations between variables.

**'performance_test**': Performance on the test set (if a test set was defined)

**'model_interpretation_natural_language'**: String, description how well the model performed.

'**shapley_summary**': String, natural language description of the most negative and positive variables.

'**successful_execution**': Bool, if False, then the algorithm needs to be re-executed. If the algorithm fails everytime, then there is aproblem with the dataset or the parameters.

'**issues_log**': dict, Reports issues that exist in the data


​    

## High level description of the run_all() function



1. Read the data, and perform some basic data cleaning.
2. Define or infer the task (classification/regression)
3. Fix skewed target variable by applying a natural logarithm function (if fix_skew is True)
4. Run the equation model
5. Determine whether execution was successful or not. Unsuccessful execution means that either we need to increase the parameters (more generations, larger populations, etc.), or that simply the dataset is very bad.
6. Determine which are the most important variables and produce the results
7. Extract patterns.
8. Run causal analysis.



# Equation model

This is the core of the ADAN engine and the run_all() function.

1. Use a genetic programming algorithm in order to extract useful features from the dataset. The algorithm takes the original features, and then produces combination of features by using operators. For example, it might create featuers like log(x1)/(x2+x3). The features are scored on how well they correlate with the target variable, and how low the correlation is with each other. We want features with high correlation with the target variable, and low correlation in between them. This part returns a number of N features.
2. The N features remaining are then used to create a regression model. A genetic algorithm tries to find the best subset of features, by using them in an elastic net and regression model, with cross-validation using cv=3. This step produces multiple different models for different values of alpha (elastic net)
3. At this step we are using a pareto optimality criterion, in order to find the best equation in terms of complexity and performance. We keep all the equations on the pareto frontier, and use the one with the best performance as the final one.
4. At this step we run some additional tests. Some times we get numerical errors, when comparing results between sklearn and the model. In some cases, due to the symbolic regression containing terms like Exp, etc. the results might be NaN. In those cases, ADAN is return the mean (regression) or the mode (classification) of the target variable. The protocol will try out different models in order to fix this, but there is a chance this might fail.
5. Final model is returned, alongside lots of diagnostic information.





### Model

Attributes

### Model

**Attributes**

model: The equation model in simply. Use str(model) to convert it into text

performance: The performance of the model during the cross-validation stage.



**Functions**

    def evaluate(self,df_or_datapoint,convert_class_output=False,fix_margins_low=False,fix_margins_high=False)
      """
      df_or_datapoint: either a dataframe, formatted in exactly the same way
      as the original dataframe, or a dictionary (which will then be converted 
      into a dataframe internally)
    
      convert_class_output: If True, then the returned result will be a pandas Series object (if regression) with predictions for each row
      or a Pandas dataframe oobject (if classification) with probabilities for each category
    
      If false, then the results will be numpy arrays. The former should be used for displaying
      output in the frontend. The latter shouldbe used for handling metrics calculation internally.
    
      fix_margins_low/high: If True, then the predicted values will never be higher or lower than the
      maximum and minimum values that existed in the original dataset.
    
      """
returns: A dataframe with the response



`def return_variables(self)`

returns: List of the variables used by the model



    def explain(self,df_original,plot=False):
        """
        Uses a decision tree surrogate model, alongside shapley values in order
        to explain the importance of the various features.
            
        Warning: In case ADAN produces a model that does not predict every category (e.g.
        there are 3 categories, but the model predicts always categories 1 or 3), then shapley values
        will return with less categories than the ones that exist in the original dataset.
        """
​    returns: A single dataframe for regression, multiple dataframes for classification.