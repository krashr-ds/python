#
# K. Rasku RN BSN & Programmer / Analyst
# My python data science library
# Initiated 1/29/2020
#
# LEARN.py
#
#### Model training
#### Model testing
#### Pipeline creation
#
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.base as skbase
import sklearn.metrics as smet
import sklearn.pipeline as skpipe
import sklearn.preprocessing as skpre
import sklearn.linear_model as lm
import sklearn.model_selection as ms

# from Hands On ML Book
# HousingAttributesAdder
#
class HousingAttributesAdder(skbase.BaseEstimator, skbase.TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["rooms_per_household"] = X["total_rooms"] / X["households"]
        X["population_per_household"] = X["population"] / X["households"]
        if self.add_bedrooms_per_room:
            X["bedrooms_per_room"] = X["total_bedrooms"] / X["total_rooms"]
        # print(X.columns)
        return X

# These two plotting functions by Kevin Arvai github.com/arvkevi
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.figure(figsize=(8, 8))
    plt.title("Precision and Recall Scores as a function of the decision threshold")
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')


def plot_roc_curve(fpr, tpr, label=None):

    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')


# Plot data columns to see distribution before scaling
#
def plot_for_scale_type(df, cols):
    for c in cols:
        plt.title = "Before scaling: "
        sb.kdeplot(df[c])
        plt.show()

# Train and test a Logistic Regression model for a large data set
# using SGDClassifier, loss="log"
#
def initTrainLargeLogistic(x, y, test_x, test_y):

    # NOTE: Always examine and scale data prior to LR
    # Using SGD on Logistic Regression Model for performance
    logModel = lm.SGDClassifier(loss="log", penalty="l2")

    # Cross Validate using 10 iterations
    scores = ms.cross_val_score(logModel, x, y, cv = 10, scoring = 'f1_weighted')
    print("Initial accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # Evaluate hyperparameters (but do not change loss type)
    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "penalty": ["l2", "l1", "elasticnet", "none"],
    }

    logModel2 = lm.SGDClassifier(loss="log")
    grid = ms.GridSearchCV(logModel2, param_grid=params, cv=10)
    grid.fit(x, y)
    print("Params: ")
    print(grid.best_params_)


# Show all Model metrics
# And plot PR and ROC curves
#
def showMetrics(model, X, y, yTest, yPredicted, cv=5):

    # Confusion Matrix
    cm = smet.confusion_matrix(yTest, yPredicted)
    print("Confusion Matrix: ")
    print("Row 1 - Things that were NOT 5s: true negatives, false positives")
    print("Row 2 - Things that were 5s: false negatives, true positives")
    print(cm)
    print("\n")

    # Classification Report - Precision, Recall, F1 Score, Support
    #
    # Precision (TPR) = true positives / (true positives + false positives)
    # Recall (Sensitivity) = true positives / (true positives + false negatives)
    # F1 Score: harmonic mean of precision and recall
    # Obviously favors classifiers with similar precision & recall (not always the goal)
    # Increasing precision reduced recall (precision / recall trade-off)
    # F1 = (2 / (1/precision + 1/recall)) = tp / tp + ((fn + fp)/ 2)
    #
    cr = smet.classification_report(yTest, yPredicted)
    print(cr)
    print("\n")

    # Full Decision Function Analysis for model
    y_scores =ms.cross_val_predict(model, x, y, cv=cv, method="decision_function")
    precisions, recalls, thresholds = smet.precision_recall_curve(y, y_scores)
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()

    # The ROC: sensitivity (recall) vs. 1 - specificity (TNR)
    fpr, tpr, thresholds = smet.roc_curve(y, y_scores)
    plot_roc_curve(fpr, tpr)
    plt.show()

    # Compute AUC
    auc_score = smet.roc_auc_score(y, y_scores)
    print("AUC Score: ")
    print(auc_score)
    print("\n")

    # If the positive class is rare, and you want to avoid false
    # positives more than false negatives - prefer PR curve.
    # Otherwise, use AUC.



