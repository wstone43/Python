import pandas as pd
import matplotlib.pyplot as plt
import os 
import csv
from datetime import datetime
import numpy as np  
from sklearn.datasets import make_regression
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 1000)

tsla = 'c:/Users/wston/Documents/tsla_full.csv'
vix = 'c:/users/wston/Documents/vix_History.csv'
tsla1 = 'c:/users/wston/Documents/tsla1.csv'
vix1 = 'c:/users/wston/Documents/vix1.csv'

tsla_prof={'Date':[],'Close':[]}
vix_prof={'Date':[],'Close':[]}

with open(tsla, newline='') as tsla_read:
    tsla_reader = csv.reader(tsla_read, delimiter=',')

    tsla_header = next(tsla_reader)
    print(f"CSV Header: {tsla_header[0::1]}")

    with open(tsla1, mode='w') as tsla_out:
        tsla_writer = csv.writer(tsla_out)
        tsla_prof = {jump[0]:jump[1] for jump in tsla_reader}

        for i,k in tsla_reader:
            tsla_prof['Date'].append(
    	        {
                    (i[0:]),
                    
                }
        )  
            tsla_prof['Close'].append(
                {
                    (k[::1])
                    
                }
                    
        ) 

t_chunklen = 1
t_list = list(tsla_prof.items())
td = [dict(t_list[i:i + t_chunklen]) for i in range(0, len(t_list), t_chunklen)]


with open(vix, newline='') as vix_read:
    vix_reader = csv.reader(vix_read, delimiter=',') 

    vix_header = next(vix_reader)
    print(f"CSV Header: {vix_header[0::1]}")

    with open(vix1, mode='w') as vix_out:
        vix_writer = csv.writer(vix_out)
        vix_prof = {roll[0]:roll[1] for roll in vix_reader}

        for x,y in vix_reader:
            vix_prof['Date'].append(
    	        {
                    (x[0:]),
                    
                }
        )  
            vix_prof['Close'].append(
                {
                    (y[::1])
                    
                }
        ) 
        
v_chunklen = 1
v_list = list(vix_prof.items())
vd = [dict(v_list[l:l + v_chunklen]) for l in range(0, len(v_list), v_chunklen)]


t_df = pd.DataFrame(t_list)
v_df = pd.DataFrame(v_list)

t_df.columns = ['Date', 'Close']
v_df.columns = ['Date', 'Close']

t_df['Date'] = pd.to_datetime(t_df["Date"])
t_df['Close'] = t_df['Close'].astype(float)
t_df['Close'].astype(int)

v_df['Date'] = pd.to_datetime(v_df["Date"])
v_df['Close'] = v_df['Close'].astype(float)
v_df['Close'].astype(int)
 
t_df.set_index('Date', inplace=True)
v_df.set_index('Date', inplace=True)

t_df['Delta'] = t_df['Close'].pct_change(periods=-1)*100
v_df['Delta'] = v_df['Close'].pct_change(periods=-1)*100

m_delta = pd.merge(t_df['Delta'],v_df['Delta'],how='inner',left_index=True,right_index=True)

bingo = plt.plot(m_delta.index,m_delta['Delta_x'],m_delta['Delta_y'])
plt.legend(iter(bingo),('TSLA', 'VIX'))
plt.show()

m_df= pd.merge(t_df['Close'],v_df['Close'],how='inner',left_index=True,right_index=True)
m_df
df_reg = m_df[['Close_x','Close_y']].copy()
df_reg.dropna(axis='columns')
df_reg.reset_index(drop=True, inplace=True)
df_reg

regr = linear_model.LinearRegression()

X=df_reg[['Close_x']]
y=df_reg[['Close_y']]

regressor = LinearRegression()
regressor.fit(X, y)

y_pred = regressor.predict(X)

plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('mark1 vs mark2')
plt.xlabel('mark1')
plt.ylabel('mark2')
plt.show()

print(df_reg)



