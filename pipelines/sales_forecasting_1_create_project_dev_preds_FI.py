#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import random
from random import randrange
from datetime import date, datetime
import os

import sklearn
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication
from truera.client.truera_authentication import BasicAuthentication
from truera.client.ingestion import ColumnSpec, ModelOutputContext

from truera.client.ingestion.util import merge_dataframes_and_create_column_spec

# connection details
TRUERA_URL = os.environ.get['URL']
AUTH_TOKEN = os.environ.get['AUTH_TOKEN']

# Python SDK - Create TruEra workspace
auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

#create project
version=randrange(1000)
project_name = "Sales Forecasting Monitoring {} {}".format(date.today(), version)
tru.add_project(project_name, score_type="regression")
#tru.activate_client_setting('create_model_tests_on_split_ingestion')

X_train_pre = pd.read_csv('./pre_train.csv',index_col=0).reset_index()
X_train_post = pd.read_csv('./post_train.csv',index_col=0).reset_index()
y = pd.read_csv('./labels_train.csv',index_col=0).reset_index()

#for mapping post-transform features to pre
#excluding for monitoring purposes
'''FEATURE_MAP = {}
for post in X_train_post.drop(columns=['index','datetime']).columns:
    mapped = None
    for pre in X_train_pre.columns:
        if post.startswith(pre) and (mapped is None or len(mapped) < len(pre)):
            mapped = pre
    if mapped not in FEATURE_MAP:
        FEATURE_MAP[mapped] = []
    FEATURE_MAP[mapped].append(post)'''

tru.add_data_collection("OJ Sales Data")

random_forest = pickle.load(open("rf.pkl", 'rb'))
model_name = 'Random Forest Regressor'
tru.add_python_model(model_name, random_forest)

data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime', #optional for pre-prod data
                        pre_data=X_train_pre,
                        #post_data=X_train_post, #use w/feature map to reverse feature encoding
                        labels=y_df)

tru.add_data(
        data_split_name='baseline data',
        data=data_df,
        column_spec=column_spec)

# # Compute & Upload dev split Feature Influences using TruEra QII
## by default, the following function will compute predictions, feature influences, and error influences for all models and splits in the project. Params exist to constrain to specific calculations, as well as specific models or data splits
tru.compute_all()
