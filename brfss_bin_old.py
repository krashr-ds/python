# BRFSS Assignment: HDS 805
# Author: KPR  Time Frame: January 2021
#
# brfss_binary.py
# Design, apply & test binary classification models
#
# Model 1: Logistic regression
# Model 2: Naive Bayes
# Model 3: KNN
# Model 4: RF
# Model 5: Gradient Boosting
# Model 6: XGBoost


import enrich
import learn

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

import sklearn.metrics as smet
import sklearn.linear_model as lm


brfss = pd.read_csv('BRFSS_Clean_Combo.csv')

# MODEL 1: LOGISTIC REGRESSION
# ----------------------------
# prepare for logit modeling by dropping label columns
brfss1 = brfss.drop(columns=['L_STATENM', 'L_STATEAB', 'L_ASTHMA', 'L_CANCER', 'L_CHCCOPD', 'L_ADDEPEV', 'L_DIABETE',
                             'L_HEART', 'L_SMOKER', 'L_HLTHPLN', 'L_COUPLED', 'L_COMORB', 'L_AGEG5YR', 'L_SEX',
                             'L_IMPRACE', 'L_RACE', 'L_EMPLOY1', 'L_MARITAL', 'L_INCOMG', 'L_EDUCAG', 'L_BMI5CAT',
                             'L_IMONTH'], axis=1)

# print correlation matrix
# print(brfss1.corr()["COMORB_1"].sort_values(ascending=False))

# define predictors, retain y = COMORB_1
# don't use synonymous columns (things that make COMORB_1 what it is)
brfss1 = pd.DataFrame(brfss1,
                      columns=["EMPLOY1", "B_HEART", "_AGEG5YR", "_BMI5CAT", "SEX", "_SMOKER3", "B_HLTHPLN",
                               "_INCOMG", "B_COUPLED", "_EDUCAG", "COMORB_1"])

# translate NaN, blank and 9 values for non-binary categorical variables to zero
brfss1 = enrich.replaceCVs(brfss1, ["SEX", "_AGEG5YR", "EMPLOY1", "_INCOMG", "_EDUCAG", "_BMI5CAT", "_SMOKER3"],
                           [np.nan, 9, ""], 0)

# Examine data to determine scale-type
# NOTE: Commented out after use, so it doesn't show plots each time.
# learn.plot_for_scale_type(brfss1, ["SEX", "_AGEG5YR", "_IMPRACE", "_INCOMG", "_EDUCAG", "MARITAL", "EMPLOY1"])

# Data is not normally distributed; apply MinMaxScaler to non-binary columns
brfss1 = enrich.normalize(brfss1, ["SEX", "_AGEG5YR", "EMPLOY1", "_INCOMG", "_EDUCAG", "_BMI5CAT", "_SMOKER3"])


# Stratified Shuffle Split based on the outcome variable
brfss_train, brfss_test = enrich.split_train_test_strat(brfss1, "COMORB_1", 1, 0.2, 42)

# Logistic Regression - initial & GridSearch evaluation, uses StratifiedKFold for k=10
brfss_x, brfss_y, test_x, test_y = enrich.prepareXYSets(brfss_train, brfss_test, "COMORB_1")
learn.initTrainLargeLogistic(brfss_x, brfss_y, test_x, test_y)

# Params result - alpha: 0.0001, penalty: "l2"; final accuracy 0.63
logModel = lm.SGDClassifier(loss="log", penalty="l2", alpha=0.0001)
logModel.fit(brfss_x, brfss_y)
yPredicted = logModel.predict(test_x)
print('Accuracy: {:.2f}'.format(smet.accuracy_score(test_y, yPredicted)))

# Show all metrics
learn.showMetrics(logModel, brfss_x, brfss_y, test_y, yPredicted, cv=10)
