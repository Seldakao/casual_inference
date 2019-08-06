# -*- coding: utf-8 -*-
import pandas as pd

import numpy as np

from sklearn.base import TransformerMixin

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

def bucket_your_categorical(Series, threshold = 0.05):
    '''Bucket the categorical variables according to a threshold'''
    
    vcounts_fraction = Series.value_counts()/Series.value_counts().sum()
    #print(vcounts_fraction)
    to_be_replace_list = list(vcounts_fraction[vcounts_fraction < threshold].index)
    #print(to_be_replace_list)
    Series = Series.apply(lambda x: 'Other' if x in to_be_replace_list else x)
    return Series


def add_dummies(df, list_of_dummy):
    '''transform the column in the list_of_dummy into dummy variables, and concat with the original data frame'''
    for column in list_of_dummy:
        dummies = pd.get_dummies(df[column], prefix = column)
        df = df.drop(columns = column)
        df = pd.concat([df, dummies], axis=1)
    
    return df
