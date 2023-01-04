import os, sys
lib_path = os.path.abspath('/Users/stelios/Dropbox/ADAN/adan')
sys.path.append(lib_path)

import adan


import pandas as pd
import os
from os.path import isfile,join
from flask import Flask, request, redirect, url_for,render_template,json as fjson,send_from_directory
from werkzeug import secure_filename
from random_words import RandomWords
from adan.aiem.symbolic_modelling import findSymbolicExpression
from adan.aiem.genetics.genetic_programming import findFeaturesGP


df=pd.read_csv('breast_cancer_demo_adan.csv')
target='class'
features=findFeaturesGP(df=df,targets=target,ngen=3,max_tree=3,population=3000,features=20,task="classification",n_processes=1)