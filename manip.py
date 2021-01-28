#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/4/2021
# Last Updated 1/24/2021
#
# MANIP.py
#
#### regex
#### counting words, characters & null values
#### substitutions / stripping
#### scraping your own data
#### sub-setting
#### generators
#### decorators
#### date aggregation
#### dataframe aggregation and manipulation
#

# This library requires the use of the following python libraries.
# Before running this code, please make sure they are installed.
#
import numpy as np
import pandas as pd
import datetime
import re


# dummy decorator
# allows execution of any function passed to it, func
#
def do_func(func):
    def wrapper(a):
        return func(a)

    return wrapper


# read_any
# generator - read any-sized file, fhandle in blocks
# default size block = 10,000
#
def read_any(fhandle, size=10000):
    block = []
    for i in fhandle:
        block.append(i)
        if len(block) == size:
            yield block
            block = []

    if block:
        yield block


# block_operator
# open and read any-sized file located at path
# and calls an operational decorator to execute block_op_function
# on the file block
# default block size = 10,000
#
def block_operator(path, block_op_function, size=10000):
    with open(path) as lfile:
        for b in read_any(lfile, size):
            # do something, for example print
            block_op_function(b)


# count_matches
# counts all the matches in a file, f for any regex, using findall
# provide the regular expression with r'[match]' syntax
# returns: the number of matches, an integer
# requires: re
#
def count_matches(f, regex):
    fh = open(f, "r")
    file_data = fh.read()
    matches = re.findall(regex, file_data)
    return len(matches)


# sub_and_write
# substitutes all instances of a matched regex with replacement text
# writes the changes to a new file called [f]_new.[same suffix]
# provide the regular expression with r'[match]' syntax
# requires: re
#
def sub_and_write(f, regex, replacement):
    fh = open(f, "r")
    file_data = fh.read()
    new_data = re.sub(regex, replacement, file_data)
    print(new_data)
    print("\n")

    filename_list = f.split(".")
    nfnp = filename_list[0]
    nfn = nfnp + "_new." + filename_list[1]

    nfnh = open(nfn, "w")
    nfnh.write(new_data)

    # check (debug; small files only!)
    # block_operator(nhn, do_func(print))


# count
# counts file-related things: all, words or lines for a file f
# returns: an integer
#
def count(f, count_type=all):
    fh = open(f, "r")
    file_data = fh.read()
    if count_type == "word":
        return len(file_data.split(" "))

    elif count_type == "line":
        return len(file_data.split("\n"))

    else:
        # entire document / all characters
        return len(file_data)


# scrape a list of fields and their values off a web page
# or a series of web pages
# creates: a csv file
# returns: the name of the csv file written, a string
#
# def scrape_a_set(url_list, field_list):


# sub-setting
# set_names, and a dictionary of what belongs in what set
# returns: a dataframe
# requires: pandas
#
# def sub_set(set_names, set_descriptions):


# agg_dates
# bucketize dates by week, month, quarter, year, or fiscal year
# returns: a dictionary
# requires: datetime
#
# def agg_dates(agg_type, date_list, date_format_string):
#
#    date_dictionary = dict.fromkeys(['week', 'month', 'quarter', 'year', 'fiscal year'])
#    if agg_type == "week":

#    elif agg_type == "month":

#    elif agg_type == "quarter":

#    elif agg_type == "year":

#    else:
# fiscal year

#    return date_dictionary


# stack_dataframes
#     files_list: a list of csv files to convert to dataframes and stack
#     diff_column_name: the name of a column to create to differentiate the three data-sets in the new stacked dataframe.  Example: YEAR, SETNO
#     diff_value_expr: a regular expression that, when applied to the file name, finds the value to be applied to the new diff_column_name.
#                      Example: [0-9]+ to find a number in the name of a file ("mydata2017.csv")
#     orientation: If 0 (default), stack data on to of each other (height-wise) or zero axis, or if 1, stack data side by side (length-wise) or 1 axis.
#     add_index: If 0, add no index; if 1 (default) add an index and a unique ID combining the index and the value of diff_column_name
#     returns:  a big or wide dataframe
#     requires: pandas
#
def stack_dataframes(files_list, diff_column_name, diff_value_expr, orientation=0, add_index=1):
    dframes = {}
    for f in files_list:
        dframes[f] = pd.read_csv(f)
        dframes[f][diff_column_name] = re.findall(diff_value_expr, f)[0]
        if add_index:
            dframes[f]["Index"] = dframes[f].index
            dframes[f]["ID"] = dframes[f][diff_column_name].map(str) + dframes[f]["Index"].map(str)

    return pd.concat(dframes, orientation)


# count_missing
# counts missing values in a dataframe (df), or counts only values of the columns requested by (cols)
# if orientation is 0 (default) compute counts col-wise, otherwise compute them row-wise (orientation=1)
# returns: Series
# requires: pandas
#
def count_missing(df, cols=[], orientation=0):
    if len(cols) > 0:
        # subset first
        df = pd.DataFrame(df, cols)

    # count everything
    if orientation:
        return df.isna().sum(axis=1)
    else:
        return df.isna().sum()


# percent_missing
# produces a dataframe of the percentage of missing values for each column of a dataframe
# returns: Series
# requires: pandas
#
def percent_missing(df):
    return df.isna().mean().round(4) * 100


# combine_columns; transform a dataframe
#   combine_list: two-element list of names of cols that will be combined
#   new_col: name of the new column that will be added to the dataframe
#            containing the combined values of combine_list
#   requires: pandas
#
def combine_columns(df, combine_list, new_col):
    if len(combine_list) == 2:
        df[new_col] = df[combine_list[0]].combine_first(df[combine_list[1]])


# combine_by_re; transform a dataframe
#   col_expr: a regex that matches the name(s) of the columns to combine
#   new_col: name of the new column that will be added to the dataframe
#            containing the combined values of m_cols
#   requires: pandas, re
#
def combine_by_re(df, col_expr, verbose=True):
    # faster execution time than list(df)
    all_cols = df.columns.values.tolist()
    m_cols = re.findall(col_expr, " ".join(all_cols))

    col_root = col_expr.split('[')[0]
    new_col = ""
    if len(re.findall(r'^_', col_root)) > 0:
        new_col = "N" + col_root
    else:
        new_col = "N_" + col_root

    if len(m_cols) == 2:
        if verbose:
            print("Combining")
            print(m_cols)
            print("Into")
            print(new_col)
        combine_columns(df, m_cols, new_col)
    elif len(m_cols) == 1:
        # don't do anything, nothing to combine
        return
    else:
        # too many things to combine
        print("Unable to combine ")
        print(m_cols)


# create_binary; add a new binary column to a dataframe based on the values in another column
#   cols: one or more existing columns to be mapped to the new binary column
#   yes_values: a list of the values of the existing column(s) to be associated with the value 1, or True
#   no_values: a list of the values of the existing column(s) to be associated with the value 0, or False
#   new_col: the name you want to give to the new binary column example) B_DIABETES
#   requires: numpy
#
def create_binary(df, cols, yes_values, no_values, new_col):
    for c in cols:
        cur_values = [df[c].isin(yes_values), df[c].isin(no_values)]
        new_values = [1, 0]
        df[new_col] = np.select(cur_values, new_values)


# recode_col: create a new_col in dataframe (df), translating the values of existing_col
# (keys of mapping_dict), to new values (values of mapping_dict)
# example mapping_dict: { 1 : 'Alabama', 2 : 'Alaska', 4 : 'Arizona', 5 : 'Arkansas', 6 : 'California', 8 : 'Colorado', 9: 'Connecticut'}
# requires: pandas
#
def recode_col(df, existing_col, mapping_dict, new_col):
    df[new_col] = df[existing_col].map(mapping_dict)


# numeric_only: return a new dataframe that is a subset of the dataframe argument (df)
# containing only continuous variables.
# This only works if categorical variables and binary variables are NOT coded.
# requires: pandas
#
def numeric_only(df):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    return df.select_dtypes(include=numerics)
