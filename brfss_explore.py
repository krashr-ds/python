import utils
import manip
import enrich
import learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

med_df = pd.read_csv("BRFSS_Clean_Combo.csv")

# Plot histograms
for c in ["B_ASTHMA", "B_CANCER", "B_CHCCOPD", "B_ADDEPEV", "B_DIABETE", "B_HEART",
          "TOTCHRONIC", "_AGEG5YR", "_SEX2", "_IMPRACE", "_INCOMG", "_EDUCAG", "MARITAL",
          "EMPLOY1", "_BMI5CAT", "_SMOKER3"]:
    med_df[c].hist()
    plt.show()



