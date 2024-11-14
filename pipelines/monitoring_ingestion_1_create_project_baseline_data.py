#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
import random
from random import randrange
from datetime import date, datetime

import sklearn
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

#load data
train_df = pd.read_csv('./split_sim_v1/train_df.csv',index_col=0).reset_index()
X_train_post = pd.read_csv('./post_train.csv',index_col=0).reset_index()
y = pd.read_csv('./labels_train.csv',index_col=0).reset_index()

#create 'data collection' in TruEra project, for data
tru.add_data_collection("OJ Sales Data")

#add model to project
random_forest = pickle.load(open("rf.pkl", 'rb'))
model_name = 'Random Forest Regressor'
tru.add_python_model(model_name, random_forest)

#prepare data - truera SDK convenience function to merge and create column specification
data_df, column_spec = merge_dataframes_and_create_column_spec(
                        id_col_name='index',
                        timestamp_col_name='datetime', #optional for pre-prod data
                        pre_data=X_train_pre,
                        labels=y_df)

#add data to project
tru.add_data(
        data_split_name='baseline data',
        data=data_df,
        column_spec=column_spec)

#note: use truera-qii if possible/truera-qii installed. Otherwise, omit this setting; TruEra will use the OSS SHAP library that corresponds to your model and prediction type. Be aware that this may lead to lengthy increases in computation time to generate Shapley value estimates.
tru.set_influence_type('truera-qii')

# # Compute & Upload dev split Feature Influences using TruEra QII
## by default, the following function will compute predictions, feature influences, and error influences for all models and splits in the project. Params exist to constrain to specific calculations, as well as specific models or data splits
tru.compute_all()

# Save artifacts for future use




### Next

script 2: add prod data
- set existing project context
- load prod data (and predictions)
- add prod data

script 3: generate feature influences for prod data
- create time range split
- generate feature influences for time range split
- perform analysis (programatically;intro to explainers)
