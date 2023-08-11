import os
import csv
import pandas as pd

ac='c:/users/wston/documents/jets.csv'
ap='c:/users/wston/documents/airport.csv'

path1=pd.read_csv(ac)
path2=pd.read_csv(ap)

df_ap=pd.DataFrame(path1)
df_ac=pd.DataFrame(path2)
print(df_ap.head())
print(df_ac.head())