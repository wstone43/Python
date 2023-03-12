from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from itertools import chain
import sklearn.metrics as metrics
import pandas as pd 
import numpy as np

file1 = 'c:/users/wston/documents/python/pybuild/TSLA.csv'

jump = pd.read_csv(file1)
datetime_series = pd.DatetimeIndex(jump['Date'])
datetime_series
jump = jump.set_index(datetime_series)

num_obs = jump[['Close']].count()
num_obs


def add_memory(s,n_days=50,mem_strength=0.1):
    ''' adds autoregressive behavior to series of data'''
    add_ewm = lambda x: (1-mem_strength)*x + mem_strength*x.ewm(n_days).mean()
    out = s.groupby(level='Date').apply(add_ewm)
    return out

# generate feature data
f01 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f01 = add_memory(f01,10,0.1)
f02 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f02 = add_memory(f02,10,0.1)
f03 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f03 = add_memory(f03,10,0.1)
f04 = pd.Series(np.random.randn(3182),index=jump[['Close']].index)
f04 = f04 # no memory

features = pd.concat([f01,f02,f03,f04],axis=1)

outcome =   f01 * np.linspace(0.5,1.5,3182) + \
            f02 * np.linspace(1.5,0.5,3182) + \
            f03 * pd.Series(np.sin(2*np.pi*np.linspace(0,1,3182)*2)+1,index=f03.index) + \
            f04 + \
            np.random.randn(3182) * 3 
outcome.name = 'outcome'


recalc_dates = features.resample('D',level='Date').mean().values[:-1]
flat = list(chain.from_iterable(recalc_dates))
recalc_dates = pd.Series(flat)
models = pd.Series(index=recalc_dates)
for date in recalc_dates:
    X_train = features.iloc[0:3182]
    y_train = outcome.iloc[0:3182]
    model = LinearRegression()
    model.fit(X_train,y_train)
    models.loc[date] = model


begin_dates = models.index
end_dates = models.index[1:].append(pd.to_datetime(['2099-12-31']))

predictions = pd.Series(index=features.index)

for i,model in enumerate(models): #loop thru each models object in collection
    X = features.iloc[0:3182]
    p = pd.Series(model.predict(X),index=X.index)
    predictions.loc[X.index] = p

common_idx = outcome.dropna().index.intersection(predictions.dropna().index)
y_true = outcome[common_idx]
y_true.name = 'y_true'
y_pred = predictions[common_idx]
y_pred.name = 'y_pred'

standard_metrics = pd.Series()

standard_metrics.loc['explained variance'] = metrics.explained_variance_score(y_true, y_pred)
standard_metrics.loc['MAE'] = metrics.mean_absolute_error(y_true, y_pred)
standard_metrics.loc['MSE'] = metrics.mean_squared_error(y_true, y_pred)
standard_metrics.loc['MedAE'] = metrics.median_absolute_error(y_true, y_pred)
standard_metrics.loc['RSQ'] = metrics.r2_score(y_true, y_pred)

print(standard_metrics)
print(pd.concat([y_pred,y_true],axis=1).tail())

def make_df(y_pred,y_true):
    y_pred.name = 'y_pred'
    y_true.name = 'y_true'
    
    df = pd.concat([y_pred,y_true],axis=1)

    df['sign_pred'] = df.y_pred.apply(np.sign)
    df['sign_true'] = df.y_true.apply(np.sign)
    df['is_correct'] = 0
    df.loc[df.sign_pred * df.sign_true > 0 ,'is_correct'] = 1 # only registers 1 when prediction was made AND it was correct
    df['is_incorrect'] = 0
    df.loc[df.sign_pred * df.sign_true < 0,'is_incorrect'] = 1 # only registers 1 when prediction was made AND it was wrong
    df['is_predicted'] = df.is_correct + df.is_incorrect
    df['result'] = df.sign_pred * df.y_true 
    return df

df = make_df(y_pred,y_true)
print(df.dropna().tail())

def calc_scorecard(df):
    scorecard = pd.Series()
    # building block metrics
    scorecard.loc['accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    
    return scorecard    

calc_scorecard(df)

def calc_scorecard(df):
    scorecard = pd.Series()
    # building block metrics
    scorecard.loc['accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()

    # derived metrics
    scorecard.loc['y_true_chg'] = df.y_true.abs().mean()
    scorecard.loc['y_pred_chg'] = df.y_pred.abs().mean()
    scorecard.loc['prediction_calibration'] = scorecard.loc['y_pred_chg']/scorecard.loc['y_true_chg']
    scorecard.loc['capture_ratio'] = scorecard.loc['edge']/scorecard.loc['y_true_chg']*100

    return scorecard    

calc_scorecard(df)

def calc_scorecard(df):
    scorecard = pd.Series()
    # building block metrics
    scorecard.loc['accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()

    # derived metrics
    scorecard.loc['y_true_chg'] = df.y_true.abs().mean()
    scorecard.loc['y_pred_chg'] = df.y_pred.abs().mean()
    scorecard.loc['prediction_calibration'] = scorecard.loc['y_pred_chg']/scorecard.loc['y_true_chg']
    scorecard.loc['capture_ratio'] = scorecard.loc['edge']/scorecard.loc['y_true_chg']*100

    # metrics for a subset of predictions
    scorecard.loc['edge_long'] = df[df.sign_pred == 1].result.mean()  - df.y_true.mean()
    scorecard.loc['edge_short'] = df[df.sign_pred == -1].result.mean()  - df.y_true.mean()

    scorecard.loc['edge_win'] = df[df.is_correct == 1].result.mean()  - df.y_true.mean()
    scorecard.loc['edge_lose'] = df[df.is_incorrect == 1].result.mean()  - df.y_true.mean()

    return scorecard    

calc_scorecard(df)

def scorecard_by_year(df):
    df['year'] = df.index.get_level_values('Date').year
    return df.groupby('year').apply(calc_scorecard).T

print(scorecard_by_year(df))

def scorecard_by_symbol(df):
    return df.groupby(level='Date').apply(calc_scorecard).T

print(scorecard_by_symbol(df))

X_train,X_test,y_train,y_test = train_test_split(features,outcome,test_size=0.20,shuffle=False)

model1 = LinearRegression().fit(X_train,y_train)
model1_train = pd.Series(model1.predict(X_train),index=X_train.index)
model1_test = pd.Series(model1.predict(X_test),index=X_test.index)

model2 = RandomForestRegressor().fit(X_train,y_train)
model2_train = pd.Series(model2.predict(X_train),index=X_train.index)
model2_test = pd.Series(model2.predict(X_test),index=X_test.index)

# create dataframes for each 
model1_train_df = make_df(model1_train,y_train)
model1_test_df = make_df(model1_test,y_test)
model2_train_df = make_df(model2_train,y_train)
model2_test_df = make_df(model2_test,y_test)

s1 = calc_scorecard(model1_train_df)
s1.name = 'model1_train'
s2 = calc_scorecard(model1_test_df)
s2.name = 'model1_test'
s3 = calc_scorecard(model2_train_df)
s3.name = 'model2_train'
s4 = calc_scorecard(model2_test_df)
s4.name = 'model2_test'

print(pd.concat([s1,s2,s3,s4],axis=1))
