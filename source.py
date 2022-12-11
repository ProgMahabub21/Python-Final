import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from pandas.plotting import parallel_coordinates
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# read excel file and create pandas dataframe
df = pd.read_excel('Final-Data-COVID-PATIENT-STATE-2020-2022-dirty-data.xlsx')


# --------- data preprossesing (e.g. data cleaning) -------------


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


# after step 2 115 more data rows are dropped


# 4. drop the rows that have duplicate values

df.drop_duplicates(inplace=True)
# convert all column to integer
df['Age'] = df['Age'].astype(int)
df['S1: Fever'] = df['S1: Fever'].astype(int)
df['S2: Cough'] = df['S2: Cough'].astype(int)
df['S3: Joint Pain'] = df['S3: Joint Pain'].astype(int)
df['Test2:RT PCR'] = df['Test2:RT PCR'].astype(int)
df['Test4: CT Scan '] = df['Test4: CT Scan '].astype(int)


# print(df.head())
# print(df.info())

# Exploratory Data Analysis
print("----------------- Data Frame -----------------")
print(df.describe())
print("----------------- Group by Risk Factor -----------------")
print(df.groupby('Risk Factor').size())

# # remove unnecessary columns
df.drop(['Date of Admission ', 'Date of Discharge',
         'Date of Expired ', 'Test1: RAT'], axis=1, inplace=True)


# ---------------- Exploratory data analysis -----------------

# let's visualise the number of samples for each class with count plot
sns.countplot(x='Risk Factor', data=df)
plt.title('Number of samples for each class')
plt.show()

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


# # let's create pairplot to visualise the data for each pair of attributes
sns.pairplot(df, hue='Risk Factor', height=2.5, palette='colorblind')
plt.show()

# parallel_coordinates(df, 'Risk Factor', colormap='rainbow')

# Feature matrix

X = df[['Age', 'S1: Fever', 'S2: Cough', 'S3: Joint Pain',
        'Test2:RT PCR', 'Test4: CT Scan ']]

print("----------------- Feature Matrix -----------------")
print(X.head())

y = df['Risk Factor']
print("----------------- Target Matrix -----------------")
print(y.head())

# ------------------ Model Training ------------------


# split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.35, random_state=16)


print("X_train shape: ", X_train.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)

score = {}
# define models
model_svm = svm.SVC()
model_nb = GaussianNB()
model_knn = KNeighborsClassifier()
model_lr = LogisticRegression()
model_dt = DecisionTreeClassifier()


model_svm.fit(X_train, y_train)
model_nb.fit(X_train, y_train)
model_knn.fit(X_train, y_train)
model_lr.fit(X_train, y_train)
model_dt.fit(X_train, y_train)

y_pred_svm = model_svm.predict(X_test)
score_svm = metrics.accuracy_score(y_test, y_pred_svm).round(4)
score["SVM"] = score_svm

y_pred_nb = model_nb.predict(X_test)
score_nb = metrics.accuracy_score(y_test, y_pred_nb).round(4)
score["Naive Bayes"] = score_nb

y_pred_knn = model_knn.predict(X_test)
score_knn = metrics.accuracy_score(y_test, y_pred_knn).round(4)
score["KNN"] = score_knn

y_pred_lr = model_lr.predict(X_test)
score_lr = metrics.accuracy_score(y_test, y_pred_lr).round(4)
score["Logistic Regression"] = score_lr

y_pred_dt = model_dt.predict(X_test)
score_dt = metrics.accuracy_score(y_test, y_pred_dt).round(4)
score["Decision Tree"] = score_dt

print("----------------------------------------")
print("Accuracy score of SVM: ", score_svm)
print("----------------------------------------")
print("Accuracy score of Naive Bayes: ", score_nb)
print("----------------------------------------")
print("Accuracy score of KNN: ", score_knn)
print("----------------------------------------")
print("Accuracy score of Logistic Regression: ", score_lr)
print("----------------------------------------")
print("Accuracy score of Decision Tree: ", score_dt)
print("----------------------------------------")

print("The accuracy scores of different Models:")
print("----------------------------------------")
for key, value in score.items():
    print(key, ":", value)
