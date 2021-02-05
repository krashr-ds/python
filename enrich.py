#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/4/2020
#
# ENRICH.py
#
#### train-test preparation
#### distributions
#### statistics
#### matrix math
#### probability
#

import numpy as np
import pandas as pd
import zlib as zl

import sklearn.preprocessing as skp
import sklearn.model_selection as ms


# split a dataframe into a test set and training set
# using a random method
# supply seed / random_state for reproducibility
#
def split_train_test(data, test_ratio, seed):
    rs = np.random.RandomState(seed)
    idx_shuffle = rs.permutation(len(data))
    td_size = int(len(data) * test_ratio)
    ts_idxs = idx_shuffle[:td_size]
    tr_idxs = idx_shuffle[td_size:]
    return data.iloc[tr_idxs], data.iloc[ts_idxs]


# split a dataframe into a test set and training set
# using an index
def test_set_check(identifier, test_ratio):
    return zl.crc32(np.int64(identifier)) & Oxffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column="index"):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# split a dataframe into a test set and training set
# using randomization on strata for proportional representation
#
def split_train_test_strat(data, split_category, n_splits, test_ratio, seed):
    strat_train_set = pd.DataFrame(data=None, columns=data.columns, index=data.index)
    strat_test_set = pd.DataFrame(data=None, columns=data.columns, index=data.index)
    sd = ms.StratifiedShuffleSplit(n_splits=n_splits, test_size=test_ratio, random_state=seed)
    for train_idx, test_idx in sd.split(data, data[split_category]):
        strat_train_set = data.loc[train_idx]
        strat_test_set = data.loc[test_idx]
    return strat_train_set, strat_test_set


# create a categorical value from a continuous value
# may be used to create strata
#
def add_categorical_from_continuous(df, cont_col, cat_col_name, bins, labels):
    df[cat_col_name] = pd.cut(df[cont_col], bins=bins, labels=labels)


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


# One-Hot encode a categorical variable using pd.get_dummies
# When drop=True, drop the categorical variable you used to
# create the one-hot columns
def one_hot_encode(ds, cat_attribute_name, drop=True):
    one_hot = pd.get_dummies(ds[cat_attribute_name])
    if drop:
        ds = ds.drop(cat_attribute_name, axis=1)
    ds = ds.join(one_hot)
    return ds
