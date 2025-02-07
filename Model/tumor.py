import pandas as pd

tumor=pd.read_csv('data\\tumor.csv')

tumor=tumor.drop(columns=['Sample code number'],errors='ignore')

print(tumor.head())

print(tumor.isnull().sum())

print(tumor.info())

