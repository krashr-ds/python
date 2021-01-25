#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/4/2020
#
# ENRICH.py
#
#### statistics
#### matrix math
#### probability
#

import pandas as pd
import sklearn.preprocessing as skp
import numpy as np


# standardize dataframe of continuous variables to a new dataframe
# to index, join to original dataframe
#
def standardize(df):
    scale = skp.StandardScaler()
    return pd.DataFrame(scale.fit_transform(df))


# normalize dataframe of continuous variables to a new dataframe
# to index, join to original dataframe
#
def normalize(df):
    scale = skp.MinMaxScaler()
    return pd.DataFrame(scale.fit_transform(df))


# summarize dataframe of continuous variables to a new dataframe
# IMPORTANT: 'describe' method ignores NULL and NaN values!!
#
def summarize(df):
    return pd.DataFrame(df.describe())