#!/usr/bin/env python
# coding: utf-8

# script arguments 
## 1: project name, as string
## 2: model name, as string

### Next

script 2: add prod data
- set existing project context
- load prod data (and predictions)
- add prod data


import pandas as pd
import numpy as np
from datetime import date, datetime

from truera.client.truera_workspace import TrueraWorkspace
from truera.client.truera_authentication import TokenAuthentication

# connection details
TRUERA_URL = os.environ.get['URL']
AUTH_TOKEN = os.environ.get['AUTH_TOKEN']

# Python SDK - Create TruEra workspace
auth = TokenAuthentication(AUTH_TOKEN)
tru = TrueraWorkspace(TRUERA_URL, auth, ignore_version_mismatch=True)

import sys
projectName = str(sys.argv[1])
tru.set_project(projectName)

# ## Add production data
modelName= str(sys.argv[2])
tru.set_model(modelName)

# ## demo-specific step: check datetimes of baseline data
tru.set_data_split("baseline data")
## Monitoring: Production Data
baseline_df = tru.get_xs()

baseline_end = baseline_df.datetime.max()

#set prod start date to end of baseline data

prod_start = baseline_end

'''The following function combines many of the prior steps that were performed cell by cell, into a single call. 
All params are required:
- modelObject: the actual model file produced from training. Used to generate predictions & feature influences
- modelName: model name as added to project, in step 2.5
- start: start date, as a string, in format corresponding to that used in modeling & artifact generation process
- periodDays: convenience function in case one wants to score data on a different period. In this scenario, we set it to 1 for daily inference. This is set to correspond to existing debugging workflow for this demo scenario. 
- numDays: the number of days worth of daily inference data to partition and ingest. This is set to correspond to existing debugging workflow for this demo scenario. 
- schema: corresponds to demo-specific modeling process and corresponding dev project directory structure

Please see the ingestion_utils.py file to inspect this function's contents.''' 

'''General steps:
- Create TruEra workspace
- Based on several params:
  - partition data for production scoring period simulation
  - score & generate feature influences
  - add these model output artifacts to the existing project (to an existing split, within a data collection that corresponds to the model object in use)'''

import imp
imp.reload(ingestion_utils)

#for demo purposes -- simple way to manipulate datetime column to generate desired production interval to load
print(prod_start)
u = datetime.strptime(prod_start,'%Y-%m-%d')
d = timedelta(days=60)
prod_end = u + d
print(prod_end)

print(projectName, dcName, random_forest, model_name, prod_start, prod_end, schema_name)

ingestion_utils.add_prod_data_and_compute(projectName, 
                                          dcName, 
                                          random_forest, 
                                          model_name, 
                                          prod_start, 
                                          prod_end, 
                                          schema_name,
                                          include_FIs=False)


