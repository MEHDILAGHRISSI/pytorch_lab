import pandas as pd

# Load the data
df = pd.read_csv('../data_set/fundamentals.csv', na_values='?')

# Dataset Overview
df.head() # preview a sample

df.shape  # number of observations and features
print(f"Number of observations: {df.shape[0]}")

df.dtypes # data types
print(f"Data types:\n{df.dtypes}")
#age                int64
#workclass         object
#fnlwgt             int64
#education         object
# (...)

df[df.duplicated()] # check duplicated rows
print(f"Number of duplicated rows: {df.duplicated().sum()}")

df.isna().sum() # missing values per feature
print(f"Missing values per feature:\n{df.isna().sum()}")
#age                  0
#workclass         1836
#fnlwgt               0
# (...)

df.isna().sum().sum() # number of missing cells
print(f"Total number of missing cells: {df.isna().sum().sum()}")
round(df.isna().sum().sum() / df.size * 100, 1) # percentage of missing cells
print(f"Percentage of missing cells: {round(df.isna().sum().sum() / df.size * 100, 1)}%")