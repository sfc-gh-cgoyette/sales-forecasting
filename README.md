# sample-models: sales forecasting, orange juice

Sales Forecasting models based on:
- https://docs.microsoft.com/en-us/azure/open-datasets/dataset-oj-sales-simulated?tabs=azureml-opendatasets#columns

Data is available from:
- http://www.cs.unitn.it/~taufer/Data/oj.csv

Data Dictionary and analysis exists here: 
- http://www.cs.unitn.it/~taufer/QMMA/L10-OJ-Data.html

Resources:
1. The notebook 'sales_forecasting.ipynb' contains code for:
  a. data loading and basic edav
  b. preparing data for modeling
  c. creating simulated partitions for future monitoring purposes
  d. training two models: a ridge regression (RR) model, and a random forest regressor (RFR)
  e. creating a truera project, adding data, adding the models
  
Notes:
  - little attempt made to tune models
  - RFR has better fit on training data, but does very poorly on test split (worse than RR)
  - Demographic information was trained upon 
  - Target is the log of unit sales
  - Arbitrary 'week' timestamps were converted to 2023 dates. 

Refs for other modeling techniques in public domain:
- https://github.com/microsoft/forecasting/blob/master/examples/grocery_sales/python/00_quick_start/lightgbm_single_round.ipynb
- https://github.com/microsoft/forecasting/blob/master/examples/grocery_sales/python/02_model/lightgbm_multi_round.ipynb

