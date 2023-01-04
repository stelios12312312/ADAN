#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:56:35 2019

@author: stelios
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:52:13 2019

@author: stelios
"""

import os 
import cdt
# cdt.SETTINGS.rpath=os.environ['RSCRIPT_PATH']
#Line used by Stelios where running on his laptop
# cdt.SETTINGS.rpath='/Library/Frameworks/R.framework/Resources/RScript'
import networkx as nx
import dowhy
from dowhy import CausalModel
import pandas as pd
import numpy as np

def _causality_helper(model,data,skeleton):
    output_graph = model.predict(data, skeleton)
    matrix=nx.adjacency_matrix(output_graph).todense()
    return matrix,output_graph

def identify_causal_structure(data,target,conservative=False):
    """
    conservative: If True, then choose the causal structure with the smallest number
    of relationships.
    """
    
    #skeleton helps create a sparse graph
    glasso = cdt.independence.graph.DecisionTreeRegression()
    skeleton = glasso.predict(data)
    #print(skeleton)
    #print(nx.adj_matrix(skeleton).todense())
    
    #new_skeleton = cdt.utils.graph.remove_indirect_links(skeleton, alg='aracne')
    #print(nx.adj_matrix(new_skeleton).todense())
    
    #Try out different algorithms to convert the skeleton
    structure_algorithms=[cdt.causality.graph.GES(),cdt.causality.graph.PC()]
    rels=[]
    package=[]
    for algo in structure_algorithms:
        matrix,output_graph=_causality_helper(algo,data,skeleton)
        rels.append(matrix.sum().sum())
        package.append((matrix,output_graph))
       
    rels=np.array(rels)
    if conservative:
        minimum=np.where(rels==rels.min())[0][0]
        matrix,output_graph=package[minimum]
    else:
        maximum=np.where(rels==rels.max())[0][0]
        matrix,output_graph=package[maximum]
    
    graph_matrix=pd.DataFrame(matrix,columns=output_graph.nodes,index=output_graph.nodes)
    
    graph=" ".join(nx.generate_gml(output_graph))
    return graph_matrix, graph

def single_causal_test(data,treatment,target,graph):
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=target,
        graph=graph)
    
    # Identify causal effect and return target estimands
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    
#    # Estimate the target estimand using a statistical method.
    estimate = model.estimate_effect(identified_estimand,
                                     method_name="backdoor.linear_regression",test_significance=True)
    
    try:
        # Refute the obtained estimate using multiple robustness checks.
        refute_results = model.refute_estimate(identified_estimand, estimate,
                                               method_name='data_subset_refuter')
    except:
        print('Causal inference failed. No causal effect detected.')
        return None
    
    return identified_estimand,estimate,refute_results

def run_causal_tests(graph_matrix,data,target,graph):
    potential_causes=graph_matrix[graph_matrix[target]==1].index.tolist()
    try:
        potential_causes.remove(target)
    except:
        pass
    
    results={}
    for cause in potential_causes:
        results[cause]=single_causal_test(data,cause,target,graph)
        
    return results
    
    
def run_causality(data,target,important_variables):
    data=data.copy()
    data['y_target']=target
    data.reset_index()
    data.dropna(inplace=True)
    data=data[np.append(important_variables,'y_target')]
    matrix,graph=identify_causal_structure(data,'y_target')
    res=run_causal_tests(matrix,data,'y_target',graph)
    return res
    
def interpret_causal_results(results):
    if results is None:
        return 'No causal results identified','No significant effects identified'
    
    final_string=''
    interpret={}
    for key in results.keys():
        if results[key] is None:
            interpret[key]='no significant effect detected'
        elif results[key][1].significance_test['p_value']<0.05:
            effect=float(results[key][2].estimated_effect[0])
            if effect<0:
                effect='negative'
            else:
                effect='positive'
            interpret[key]='significant '+effect + ' effect'
    
    for key in interpret.keys():
        final_string+="The variable {0} has {1}".format(key,interpret[key])+'. '
    
    return final_string,interpret
    
#import dowhy.datasets
#data = dowhy.datasets.linear_dataset(
#    beta=10,
#    num_common_causes=5,
#    num_instruments=2,
#    num_samples=10000,
#    treatment_is_binary=True)
#
#df=data["df"]
#outcome=data["outcome_name"]

#res=run_causality(df,outcome)
    
#crime=pd.read_csv('peirama_me_crime.csv')
#target=crime['target']
#crime.drop('target',inplace=True,axis=1)
#crime.columns=['racePctWhite', 'nonViolPerPop', 'PctKids2Par',    'PctFam2Par', 'PctTeen2Par']
#res=run_causality(crime,target,['racePctWhite', 'nonViolPerPop', 'PctKids2Par',    'PctFam2Par', 'PctTeen2Par'])
#
#for key in res.keys():
#    print('********VARIABLE*******')
#    print(key)
#    print('***********************')
#    for element in res[key]:
#        print(element)
#        
#print(interpret_causal_results(res))
