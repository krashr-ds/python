import utils
import manip
import numpy as np
import pandas as pd

# Stack DataFrames
big_df = manip.stack_dataframes(["LLCP2017.csv", "LLCP2018.csv", "LLCP2019.csv"], "YEAR", "[0-9]+")


# Combine columns using Regular Expressions
# Great regex debugger: https://regex101.com/
print("Combine Columns:")

ccol_exprs = ['ALCDAY[0-9]', 'AVEDRNK[0-9]', 'DIABAGE[0-9]', 'DRNKDRI[0-9]', 'FALLINJ[0-9]', 'SLEPTIM[0-9]',
              'HEIGHT[0-9]', 'NUMPHON[0-9]', 'WEIGHT[0-9]', 'CHKHEMO[0-9]', 'MARIJAN[0-9]', 'NUMBURN[0-9]',
              'HTIN[0-9]', 'HTM[0-9]', 'WTKG[0-9]', '_BMI[0-9]', 'ADDEPEV[0-9]', 'ASTHMA[0-9]', 'CHCCOPD[0-9]',
              'CHCKDNY[0-9]', 'CSTATE[0-9]', 'CTELENM[0-9]', 'CTELENUM[0-9]', 'CVDCRHD[0-9]', 'CVDINFR[0-9]',
              'DIABETE[0-9]', 'EXERANY[0-9]', 'FLUSHOT[0-9]', 'HADHYST[0-9]', 'HADPAP[0-9]', 'HAVARTH[0-9]',
              'HLTHPLN[0-9]', 'NUMHHOL[0-9]', 'PCPSAAD[0-9]', 'PCPSADI[0-9]', 'PCPSARE[0-9]', 'PNEUVAC[0-9]',
              'PSATEST[0-9]', 'PVTRESD[0-9]', 'STATERE[0-9]', 'STOPSMK[0-9]', 'VETERAN[0-9]', 'CAREGIV[0-9]',
              'CHECKUP[0-9]', 'RSPSTAT[0-9]', 'EMPLOY[0-9]', 'IMFVPLA[0-9|A-Z]', 'LASTDEN[0-9]', 'LASTPAP[0-9]',
              'LASTSMK[0-9]', 'PERSDOC[0-9]', 'RENTHOM[0-9]', 'RMVTETH[0-9]', 'SEX[0-9]', 'SMOKDAY[0-9]',
              'USENOW[0-9]', 'PREDIAB[0-9]', 'EYEEXAM[0-9]', 'HLTHCVR[0-9]', 'DELAYME[0-9]', 'CRGVREL[0-9]',
              'CRGVLNG[0-9]', 'CRGVHRS[0-9]', 'CRGVPRB[0-9]', 'USEMRJN[0-9]', 'RSNMRJN[0-9]', 'ADPLEAS[0-9]',
              'ADDOWN[0-9]', 'CNCRTYP[0-9]', 'CSRVTRT[0-9]', 'CSRVDOC[0-9]', '_PRACE[0-9]', '_MRACE[0-9]',
              '_SMOKER[0-9]', ' _RFSMOK[0-9]']

for c in ccol_exprs:
    manip.combine_by_re(big_df, c)

# create binary columns from non-binary
#
binary_cols = ["N_DIABETE", "N_ADDEPEV", "N_CHCCOPD", "N_CHCKDNY", "N_FLUSHOT", "N_HAVARTH", "N_NUMHHOL", "N_PNEUVAC",
               "N_PVTRESD"]
for c in binary_cols:
    nc = "B_" + c.split("_")[1]
    manip.create_binary(big_df, [c], [1], [2, 3, 4, 7, 9], nc)

manip.create_binary(big_df, ["_CASTHM1"], [2], [1, 7, 9], "B_ASTHMA")


# combine multiple non-binary columns into one binary column
#
cols_to_1 = ["CHCSCNCR", "CHCOCNCR"]
manip.create_binary(big_df, cols_to_1, [1], [2, 3, 4, 7, 9], "B_CANCER")
cols_to_2 = ["CVDCRHD4", "CVDINFR4", "CVDSTRK3"]
manip.create_binary(big_df, cols_to_2, [1], [2, 3, 4, 7, 9], "B_HEART")


# Creating aggregate binary column from multiple binary columns
#
cols_to_3 = ["B_ASTHMA", "B_CANCER", "B_CHCCOPD", "B_ADDEPEV", "B_DIABETE", "B_HEART"]
manip.create_binary(big_df, cols_to_3, [1], [0], "COMORB_1")

# Create summary variable, TOTCHRONIC, containing the number of chronic conditions for each participant
#
big_df["TOTCHRONIC"] = big_df["B_ASTHMA"] + big_df["B_CANCER"] + big_df["B_CHCCOPD"] + big_df["B_ADDEPEV"] + big_df[
    "B_DIABETE"] + big_df["B_HEART"]


# Create summary variable, CHRONICGRP, with a coded categorical variable based on TOTCHRONIC
# 0 = No Chronic Conditions, 1 = One Chronic Condition, 2 = Two Chronic Conditions, 3 = 3 OR MORE Chronic Conditions
#
big_df["CHRONICGRP"] = big_df["TOTCHRONIC"]
if_cc = [big_df["TOTCHRONIC"] == 0, big_df["TOTCHRONIC"] == 1, big_df["TOTCHRONIC"] == 2, big_df["TOTCHRONIC"] >= 3]
then_cc = [0, 1, 2, 3]
big_df["CHRONICGRP"] = np.select(if_cc, then_cc)


# Create new column with State Names
states_dict = {1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California', 8: 'Colorado', 9: 'Connecticut',
               10: 'Delaware', 11: 'District of Columbia', 12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho',
               17: 'Illinois', 18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana', 23: 'Maine',
               24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan', 27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri',
               30: 'Montana', 31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey', 35: 'New Mexico',
               36: 'New York', 37: 'North Carolina', 38: 'North Dakota', 39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon',
               42: 'Pennsylvania', 44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota', 47: 'Tennessee',
               48: 'Texas', 49: 'Utah', 50: 'Vermont', 51: 'Virginia', 53: 'Washington', 54: 'West Virginia',
               55: 'Wisconsin', 56: 'Wyoming', 66: 'Guam', 72: 'Puerto Rico', 78: 'Virgin Islands', 99: ''}

manip.recode_col(big_df, "_STATE", states_dict, "_STATENM")


states_abbr_dict = {1: 'AL', 2: 'AK', 4: 'AZ', 5: 'AR', 6: 'CA', 8: 'CO', 9: 'CT', 10: 'DE', 11: 'DC', 12: 'FL',
                    13: 'GA', 15: 'HI', 16: 'ID', 17: 'IL', 18: 'IN', 19: 'IA', 20: 'KS', 21: 'KY', 22: 'LA', 23: 'ME',
                    24: 'MD', 25: 'MA', 26: 'MI', 27: 'MN', 28: 'MS', 29: 'MO', 30: 'MT', 31: 'NE', 32: 'NV', 33: 'NH',
                    34: 'NJ', 35: 'NM', 36: 'NY', 37: 'NC', 38: 'ND', 39: 'OH', 40: 'OK', 41: 'OR', 42: 'PA', 44: 'RI',
                    45: 'SC', 46: 'SD', 47: 'TN', 48: 'TX', 49: 'UT', 50: 'VT', 51: 'VA', 53: 'WA', 54: 'WV',
                    55: 'WI', 56: 'WY', 66: 'GU', 72: 'PR', 78: 'VI', 99: ''}

manip.recode_col(big_df, "_STATE", states_abbr_dict, "_STATEAB")


# Combine "SEX" columns together
manip.combine_columns(big_df, ["SEX", "SEX1"], "_SEX1")
manip.combine_columns(big_df, ["_SEX", "_SEX1"], "_SEX2")

# TRIM the DataFrame to a size that will allow for faster analysis on desired columns
med_df = pd.DataFrame(big_df, columns=["B_ASTHMA", "B_CANCER", "B_CHCCOPD", "B_ADDEPEV", "B_DIABETE", "B_HEART", "COMORB_1", "TOTCHRONIC", "CHRONICGRP",
                                       "_STATE", "_STATEAB", "_STATENM", "_AGEG5YR", "_SEX2", "_IMPRACE", "_INCOMG", "_EDUCAG", "MARITAL", "HLTHPLN1", "EMPLOY1",
                                       "_BMI5CAT", "_SMOKER3"])

# Recode _SMOKER3 and HLTHPLN1 as binary (0,1)
manip.create_binary(big_df, ["_SMOKER3"], [1], [2, 3, 4, 7, 9], "B_SMOKER")
manip.create_binary(big_df, ["HLTHPLN1"], [1], [2, 3, 4, 7, 9], "B_HLTHPLN")

# Check Data Set for NULLs
print(med_df.info())
print("\n")

# count missing columns
#
print("Count of Missing Column Values:")
missing_cols_df = utils.count_missing(med_df)
print(missing_cols_df.head())
print("\n")

# count missing rows
#
print("Count of Missing Row Values:")
missing_rows_df = utils.count_missing(med_df, orientation=1)
print(missing_rows_df.head())
print("\n")

# percent missing for all columns
#
print("Percentage Missing for All Columns:")
percent_missing_df = utils.percent_missing(med_df)
print(percent_missing_df.head())
print("\n")

# create report CSV(s)
#
missing_cols_df.to_csv("Missing_Columns_Count.csv")
percent_missing_df.to_csv("Missing_Columns_Percent.csv")

# create CSV of clean DataFrame
#
med_df.to_csv(("BRFSS_Clean_Combo.csv"))
