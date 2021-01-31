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

import sklearn.pipeline as skp
import sklearn.preprocessing as skprep
import sklearn.impute as ski
import sklearn.base as skbase


# from Hands On ML Book
# AttributesAdder
# TODO: Re-attempt generic implementation with additional functions KPR 1/30/2021
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

        return X


# derived from Hands On ML Book
# Transformation Pipeline
# TODO: Re-attempt generic implementation with decorator KPR 1/30/2021
#

def CreateHousingPipeline():
    return skp.Pipeline([
        ("imputer", ski.SimpleImputer(strategy="median")),
        ("attribs_adder", HousingAttributesAdder()),
        ("std_scaler", skprep.StandardScaler()),
    ])


