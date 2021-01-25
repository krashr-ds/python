#
# Testing calls to my libraries
# K. Rasku RN BSN  Last Modified: 1/24/2021
#

import manip
import enrich
import numpy as np
import pandas as pd

manip.block_operator("SampleTextFile_1000kb.txt", manip.do_func(print))

print("\n")

print("Counting words in file:")
print("Words: ")
print(str(manip.count("SampleTextFile_1000kb.txt", "word")))
print("Lines: ")
print(str(manip.count("SampleTextFile_1000kb.txt", "line")))
print("All Characters in document: ")
print(str(manip.count("SampleTextFile_1000kb.txt")))

print("\n")
print("Use a regex to count the number of white spaces in the file:")
print(str(manip.count_matches("SampleTextFile_1000kb.txt", r'\s')))
print("Count the number of times 'er' appears at the end of a word:")
print(str(manip.count_matches("SampleTextFile_1000kb.txt", r'er\b')))

print("\n")
print("Substitute all instances of 'er' at a word-boundary with 'as'")
manip.sub_and_write("SampleTextFile_1000kb.txt", r'er\b', "as")

# NOTE: This is commented out because this implementation is not yet complete.
#
# scrape_fields_list = ['Best Bid/Ask', '1 Year Target', 'Share Volume', '50 Day Avg. Daily Volume', 'Previous Close', '52 Week High/Low', 'Market Cap', 'P/E Ratio', 'Forward P/E (1y)', 'Earnings Per Share (EPS)', 'Annualized Dividend', 'Ex-Dividend Date', 'Dividend Payment Date', 'Current Yield', 'Beta Open Price', 'Open Date', 'Close Price', 'Close Date']
# scrape_urls_list = ['http://www.nasdaq.com/symbol/aapl']
# scraped_file = manip.scrape_a_set(scrape_urls_list, scrape_fields_list)
# print("The data I scraped is in " + str(scraped_file))
# print("\n")

print("\n")
print("Stack dataframes from a list of csv files:")
print("(NOTE: Takes about 30 seconds to run.)")

# NOTE: Comment out because of execution time if no need to re-run.
# OR, FOR TESTING: Just use stack_dataframes on ONE dataframe, for brevity.
# big_df = manip.stack_dataframes(["LLCP2018.csv"], "YEAR", "[0-9]+")
#
big_df = manip.stack_dataframes(["LLCP2017.csv", "LLCP2018.csv", "LLCP2019.csv"], "YEAR", "[0-9]+")
print(big_df.head())
print("\n")
# long_df = manip.stack_dataframes(["LLCP2017.csv", "LLCP2018.csv", "LLCP2019.csv"], "YEAR", "[0-9]+", 1)
# print(long_df.head())
# print("\n")

# count missing columns
#
print("Count of Missing Column Values:")
missing_cols_df = manip.count_missing(big_df)
print(missing_cols_df.head())
print("\n")

# count missing rows
#
print("Count of Missing Row Values:")
missing_rows_df = manip.count_missing(big_df, orientation=1)
print(missing_rows_df.head())
print("\n")

# percent missing for all columns
#
print("Percentage Missing for All Columns:")
percent_missing_df = manip.percent_missing(big_df)
print(percent_missing_df.head())
print("\n")

# create report CSV(s)
#
missing_cols_df.to_csv("Missing_Columns_Count.csv")
percent_missing_df.to_csv("Missing_Columns_Percent.csv")


print("Combine Columns:")

ccol_exprs = ['ALCDAY[0-9]', 'AVEDRNK[0-9]', 'DIABAGE[0-9]', 'DRNKDRI[0-9]', 'FALLINJ[0-9]', 'SLEPTIM[0-9]', 'HEIGHT[0-9]', 'NUMPHON[0-9]', 'WEIGHT[0-9]', 'CHKHEMO[0-9]',
              'MARIJAN[0-9]', 'NUMBURN[0-9]', 'HTIN[0-9]', 'HTM[0-9]', 'WTKG[0-9]', '_BMI[0-9]',
              'ADDEPEV[0-9]', 'ASTHMA[0-9]', 'CHCCOPD[0-9]', 'CHCKDNY[0-9]', 'CSTATE[0-9]', 'CTELENM[0-9]', 'CTELENUM[0-9]', 'CVDCRHD[0-9]', 'CVDINFR[0-9]',
              'DIABETE[0-9]', 'EXERANY[0-9]', 'FLUSHOT[0-9]', 'HADHYST[0-9]', 'HADPAP[0-9]', 'HAVARTH[0-9]', 'HLTHPLN[0-9]', 'NUMHHOL[0-9]', 'PCPSAAD[0-9]', 'PCPSADI[0-9]',
              'PCPSARE[0-9]', 'PNEUVAC[0-9]', 'PSATEST[0-9]', 'PVTRESD[0-9]', 'STATERE[0-9]', 'STOPSMK[0-9]', 'VETERAN[0-9]', 'CAREGIV[0-9]',
              'CHECKUP[0-9]', 'RSPSTAT[0-9]', 'EMPLOY[0-9]', 'IMFVPLA[0-9|A-Z]', 'LASTDEN[0-9]', 'LASTPAP[0-9]', 'LASTSMK[0-9]', 'PERSDOC[0-9]', 'RENTHOM[0-9]',
              'RMVTETH[0-9]', 'SEX[0-9]', 'SMOKDAY[0-9]', 'USENOW[0-9]', 'PREDIAB[0-9]', 'EYEEXAM[0-9]', 'HLTHCVR[0-9]', 'DELAYME[0-9]', 'CRGVREL[0-9]', 'CRGVLNG[0-9]',
              'CRGVHRS[0-9]', 'CRGVPRB[0-9]', 'USEMRJN[0-9]', 'RSNMRJN[0-9]', 'ADPLEAS[0-9]', 'ADDOWN[0-9]', 'CNCRTYP[0-9]', 'CSRVTRT[0-9]', 'CSRVDOC[0-9]', '_PRACE[0-9]',
              '_MRACE[0-9]', '_SMOKER[0-9]', ' _RFSMOK[0-9]']

for c in ccol_exprs:
    manip.combine_by_re(big_df, c)


# create binary columns from non-binary
#
binary_cols = ["N_DIABETE", "N_ADDEPEV", "N_CHCCOPD", "N_CHCKDNY", "N_FLUSHOT", "N_HAVARTH", "N_NUMHHOL", "N_PNEUVAC", "N_PVTRESD"]
for c in binary_cols:
    nc = "B_" + c.split("_")[1]
    manip.create_binary(big_df, [c], [1], [2, 3, 4, 7, 9], nc)

manip.create_binary(big_df, ["_CASTHM1"], [2], [1, 7, 9], "B_ASTHMA")
print("Binary COPD:")
print(big_df["B_CHCCOPD"])
print("\n")

# combine multiple non-binary columns into one binary column
#
cols_to_1 = ["CHCSCNCR", "CHCOCNCR"]
manip.create_binary(big_df, cols_to_1, [1], [2, 3, 4, 7, 9], "B_CANCER")
cols_to_2 = ["CVDCRHD4", "CVDINFR4", "CVDSTRK3"]
manip.create_binary(big_df, cols_to_2, [1], [2, 3, 4, 7, 9], "B_HEART")
print("Binary HEART:")
print(big_df["B_HEART"])
print("\n")

# Creating aggregate binary column from multiple binary columns
#
cols_to_3 = ["B_ASTHMA", "B_CANCER", "B_CHCCOPD", "B_ADDEPEV", "B_DIABETE", "B_HEART"]
manip.create_binary(big_df, cols_to_3, [1], [0], "COMORB_1")

# Create summary variable, TOTCHRONIC, containing the number of chronic conditions for each participant
# NOTE: Added Arthritis, because this is part of CMS constellation of Chronic Conditions
#
big_df["TOTCHRONIC"] = big_df["B_ASTHMA"] + big_df["B_CANCER"] + big_df["B_CHCCOPD"] + big_df["B_ADDEPEV"] + big_df[
    "B_DIABETE"] + big_df["B_HEART"] + big_df["B_HAVARTH"]
print("Total Chronic Conditions:")
print(big_df["TOTCHRONIC"])
print("\n")

# Create summary variable, CHRONICGRP, with a coded categorical variable based on TOTCHRONIC
# 0 = No Chronic Conditions, 1 = One Chronic Condition, 2 = Two or More Chronic Conditions
#
big_df["CHRONICGRP"] = big_df["TOTCHRONIC"]
if_cc = [big_df["TOTCHRONIC"] == 0, big_df["TOTCHRONIC"] == 1, big_df["TOTCHRONIC"] >= 2]
then_cc = [0, 1, 2]
big_df["CHRONICGRP"] = np.select(if_cc, then_cc)
print("Chronic Conditions Category:")
print(big_df["CHRONICGRP"])
print("\n")

# Print crosstab
#
print(pd.crosstab(big_df["CHRONICGRP"], big_df["TOTCHRONIC"]))
print("\n")

# Create new column with State Names
states_dict = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California', 8: 'Colorado', 9: 'Connecticut',
               10: 'Delaware', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
               18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine', 24: 'Maryland', 25: 'Massachusetts',
               26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana', 31: 'Nebraska', 32: 'Nevada',
               33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
               39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
               47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia', 55: 'Wisconsin',
               56: 'Wyoming', 66: 'Guam', 72: 'Puerto Rico', 78: 'Virgin Islands', 99: ''}
manip.recode_col(big_df, "_STATE", states_dict, "_STATENM")
print("State Names:")
print(big_df["_STATENM"])
print("\n")

# Sub-set to continuous variables
# (NOTE: this df is too big because many categorical variables are coded, and many binary variables are dummy coded.)
#continuous_df = manip.numeric_only(big_df)


cont_cols = ['ALCDAY5', 'N_AVEDRNK', 'CHILDREN', 'CPDEMO1B', 'N_DIABAGE', 'DRNKDRI2', 'FALLINJ3', 'MAXDRNKS', 'SLEPTIM1', 'HEIGHT3', 'HHADULT',
            'MENTHLTH', 'NUMMEN', 'NUMWOMEN', 'N_NUMPHON', 'PHYSHLTH', 'POORHLTH', 'WEIGHT2', 'DOCTDIAB', 'CHKHEMO3', 'FEETCHK', 'DRVISITS',
            'MARIJAN1', 'NUMBURN3', 'LCSFIRST', 'LCSLAST', 'LCSNUMCG', 'CNCRAGE', 'HTIN4', 'HTM4', 'WTKG3', 'N_BMI', '_CHLDCNT']

continuous_df = pd.DataFrame(big_df, columns=cont_cols)
stand_df = enrich.standardize(continuous_df)
print("Z-Scores of Continuous Variables: ")
print(stand_df.head())
print("\n")

norm_df = enrich.normalize(continuous_df)
print("Normalized Continuous Variables: ")
print(norm_df.head())
print("\n")

# N.B. -> The describe() function ignores NULL and NaN values!
summary_df = enrich.summarize(continuous_df)
print("Summary of Not-Null Values: ")
print(summary_df.head())
print("\n")
