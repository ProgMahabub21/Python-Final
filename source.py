import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from pandas.plotting import parallel_coordinates

# read excel file and create pandas dataframe
df = pd.read_excel('Final-Data-COVID-PATIENT-STATE-2020-2022-dirty-data.xlsx')

# print the view of the dataframe
print(df.head())

# print info
print(df.info())


# cleaning the data process is done here
# it will follow four steps

# 1. drop the rows that are not needed (have missing status values) as sensitive data so we can't fill it with mean or mode
df.dropna(subset=['Status'], inplace=True)
df.dropna(subset=['Risk Factor'], inplace=True)
df.dropna(subset=['S1: Fever'], inplace=True)
df.dropna(subset=['S2: Cough'], inplace=True)
df.dropna(subset=['S3: Joint Pain'], inplace=True)
df.dropna(subset=['S4: Shortness of Breath'], inplace=True)
df.dropna(subset=['Test1: RAT'], inplace=True)
df.dropna(subset=['Test2:RT PCR'], inplace=True)
df.dropna(subset=['Test3: Chest X Ray'], inplace=True)
df.dropna(subset=['Test4: CT Scan '], inplace=True)


# after step 1 15 data rows are dropped

# 2 Cleaning wrong data format
# convert age column to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Date of Admission '] = pd.to_datetime(
    df['Date of Admission '], errors='coerce')
df['Date of Discharge'] = pd.to_datetime(
    df['Date of Discharge'], errors='coerce')
df['Date of Expired '] = pd.to_datetime(
    df['Date of Expired '], errors='coerce')
# check if age is numeric or drop the row


df.dropna(subset=['Age'], inplace=True)
df.dropna(subset=['Date of Admission '], inplace=True)

# df.dropna(s)


print(df.head())

print(df.info())

# after step 2 115 more data rows are dropped


# 4. drop the rows that have duplicate values

df.drop_duplicates(inplace=True)

# print(df.head())
# print(df.info())

# Exploratory Data Analysis
print(df.describe())
print(df.groupby('Risk Factor').size())

# let's visualise the number of samples for each class with count plot
# sns.countplot(x='Risk Factor', data=df)
# plt.title('Number of samples for each class')
# plt.show()

# calculate the correlation between variables
corr = df.corr().round(2)
print(corr)
dataplot = sns.heatmap(df.corr(), annot=True)
plt.show()

# remove redundant values
mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask=mask, annot=True)
plt.show()


# let's create pairplot to visualise the data for each pair of attributes
# sns.pairplot(df, hue='Risk Factor', height=2.5, palette='colorblind')
# plt.show()


parallel_coordinates(df, 'Risk Factor', colormap='rainbow')
