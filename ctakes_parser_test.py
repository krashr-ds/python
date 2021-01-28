
# cTAKES Parser
# With ICD-10 Code Extraction

import pandas as pd
import ctakes_parser as parser

# Leave this commented out until changes are merged to the main project
# import ctakes_parser.ctakes_parser as parser

# Import cTAKES xmi file into a dataframe using ctakes_parser
# package written by Somshubra Majumdar and Matthew Cullen
# 1/27/2021 Added a few lines of code to include the actual ICD codes
#

xmi_df = parser.parse_file('hp1.txt.xmi')
# xmi_df = ctakes_parser.parse_file('hp1.txt.xmi')
print("cTAKES DataFrame:")
print(xmi_df.head())
print("\n")
print("cTAKES SubSet: ID, cui, negated, part of speech, preferred text, true_text, refsem, scheme, textsem")
xmi_summary_df = pd.DataFrame(xmi_df, columns=['id', 'cui', 'negated', 'part_of_speech', 'preferred_text', 'true_text',
                                               'refsem', 'scheme', 'code', 'textsem'])
print(xmi_summary_df.head())
print("\n")
icd10_df = xmi_df[xmi_df["scheme"].isin(['ICD10CM', 'ICD10AM', 'ICD10AMAE', 'ICD10AE', 'MTHICD9', 'ICD9CM'])]
icd_summary_df = pd.DataFrame(icd10_df, columns=['id', 'cui', 'negated', 'part_of_speech', 'preferred_text', 'true_text',
                                               'refsem', 'scheme', 'code', 'textsem'])
print(icd_summary_df.head())
print("\n")

# For Later Use:
# parser.parse_dir(in_directory_path='notes_in/',
#                  out_directory_path='notes_out/')