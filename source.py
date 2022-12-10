import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


#read excel file and create pandas dataframe
df = pd.read_excel('Final-Data-COVID-PATIENT-STATE-2020-2022-DIRTY-DATA_2.xlsx')

#print the view of the dataframe
print(df.head())

#print info
print(df.info())


#cleaning the data process is done here 
#it will follow four steps 

#1. drop the rows that are not needed (have missing status values)
df.dropna(subset=['Status'], inplace=True)

#after step 1 15 data rows are dropped

#2 Cleaning wrong data format
#convert age column to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Date of Admission '] = pd.to_datetime(df['Date of Admission '], errors='coerce')
df['Date of Discharge'] = pd.to_datetime(df['Date of Discharge'], errors='coerce')
df['Date of Expired '] = pd.to_datetime(df['Date of Expired '], errors='coerce')
#check if age is numeric or drop the row


df.dropna(subset=['Age'], inplace=True)
df.dropna(subset=['Date of Admission '], inplace=True)
df.dropna(subset=['Test2:RT PCR'], inplace=True)
df.dropna(subset=['Test3: Chest X Ray'], inplace=True)
df.dropna(subset=['Test4: CT Scan '], inplace=True)
#df.dropna(s)

print(df.head())

print(df.info())

#df.to_excel('Final-Data-COVID-PATIENT-STATE-2020-2022-clean-data.xlsx')

#after step 2 115 more data rows are dropped


#3. replace missing/wrong values 

for x in df.index:
    for y in df.columns:
        if df.loc[x,y] == 'Yes':
            df.loc[x,y] = 'True'
        elif df.loc[x,y] == 'No':
            df.loc[x,y] = 'False'

for x in df.index:
    if df.loc[x, 'S1: Fever'] == 'No':
        df.loc[x, 'S1: Fever'] = 'False'
    elif df.loc[x, 'S1: Fever'] == 'Yes':
        df.loc[x, 'S1: Fever'] = 'True'
    
    if df.loc[x,'S2: Cough' ] == 'No':
        df.loc[x,'S2: Cough' ] = 'False'
    elif df.loc[x,'S2: Cough' ] == 'Yes':
        df.loc[x,'S2: Cough' ] = 'True'

    if df.loc[x,'S3: Joint Pain'] == 'No':
            df.loc[x,'S3: Joint Pain'] = 'False'
    elif df.loc[x,'S3: Joint Pain'] == 'Yes':
        df.loc[x,'S3: Joint Pain'] = 'True'

    if df.loc[x,'S4: Shortness of Breath'] == 'No':
        df.loc[x,'S4: Shortness of Breath'] = 'False'
    elif df.loc[x,'S4: Shortness of Breath'] == 'Yes':
        df.loc[x,'S4: Shortness of Breath'] = 'True'
    
    if df.loc[x,'Test1: RAT'] == 'No':
        df.loc[x,'Test1: RAT'] = 'False'
    elif df.loc[x,'Test1: RAT'] == 'Yes':
        df.loc[x,'Test1: RAT'] = 'True'
    
    if df.loc[x,'Test2:RT PCR'] == 'No':
        df.loc[x,'Test2:RT PCR'] = 'False'
    elif df.loc[x,'Test2:RT PCR'] == 'Yes':
        df.loc[x,'Test2:RT PCR'] = 'True'
    
    if df.loc[x,'Test3: Chest X Ray'] == 'No':
        df.loc[x,'Test3: Chest X Ray'] = 'False'
    elif df.loc[x,'Test3: Chest X Ray'] == 'Yes':
        df.loc[x,'Test3: Chest X Ray'] = 'True'

    if df.loc[x,'Test4: CT Scan '] == 'No':
        df.loc[x,'Test4: CT Scan '] = 'False'
    elif df.loc[x,'Test4: CT Scan '] == 'Yes':
        df.loc[x,'Test4: CT Scan '] = 'True'


df.to_excel('Final-Data-COVID-PATIENT-STATE-2020-2022-clean-data.xlsx')
    

#4. drop the rows that have duplicate values

df.drop_duplicates(inplace=True)


