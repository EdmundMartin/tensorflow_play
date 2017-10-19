import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

quandl.ApiConfig.api_key = 'Snm_s1oX_V-6Po7cd71X'
df = quandl.get_table('WIKI/PRICES', ticker='GOOGL')

df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume']]

df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_close'] * 100

df = df[['adj_close', 'HL_PCT', 'PCT_change', 'adj_volume']]

forecast_column = 'adj_close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_column].shift(-forecast_out)
df.dropna(inplace=True)
print(df.head())

X = np.array(df.drop(['label'],1))

X = preprocessing.scale(X)

df.dropna(inplace=True)
y = np.array(df['label'])

print(len(X), print(y))