# adan-core-datalyst

This repository contains the core of ADAN's machine learning engine. 

# Folders
## adan
This is the core of ADAN. This is broken down into the following folders:

**aiem**: Automated intelligent equation modelling
This folder is currently the core IP behind ADAN. It contains

**aipd**: Automated intelligent pattern detection.
This folder for now, doesn't contain anything beyond a single file. The vision behind this module is to contain heuristic functions for extracting insights automatically from data. Quick wins can be: 

1) Frequent itemset mining
2) Calculate correlations and return the highest ones
3) Replace correlations above, with some fancier and more useful metric (e.g. mutual information score)

This could also be used

**aipm**: Automated intelligent predictive modelling. This is pure AutoML. It uses a genetic algorithm, or a particle swarm optimisation method in combination with Bayesian optimisation (based on Gaussian processes), in order to find the optimal parameters for a model. However, this was abandoned in favour of focusing on AIEM. This kind of AutoML (optimising hyperparams) has progressed and auto-scikitlearn is the standard.

**aist**: Automated intelligent storytelling. This is used in order to provide natural language explanations of the results. So far, the 

**api**: The api.

**metrics**: Helper folder. It contains the metrics that are used for optimisation. Some metrics are using Numba for faster computation  (http://numba.pydata.org/). 


# SyntheticData
Contains the synthetic data module. Please go to that README to see more details about how it works.

# Datatests
This is the testing folder for ADAN. WARNING: this folder is a bit messy right now.
The main file is test_datasets.py. This will go through datasets in each folder (note: folders are not detecting automatically), and it will run the core AIEM protocol. 

We will keep adding more and more datasets as we go along. 

Please ignore other files in this folder.

# Installation notes

LightGBM might cause issues when installed through pip. Please follow the installation guide here: https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html or it might be better to simply install using Anaconda https://anaconda.org/conda-forge/lightgbm