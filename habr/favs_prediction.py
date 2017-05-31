
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import os
import pandas as pd

import requests
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import math
from fbprophet import Prophet
import numpy as np


def prepare_data(csv_file, window = 60):
    df = pd.read_csv(csv_file)
    df['published'] = pd.to_datetime(df['published'])
    df['favs'] = df['favs_lognorm'].apply(math.expm1)
    by_day = df[['published', 'favs']].set_index('published').resample('D').sum().fillna(0)
    by_day['favs_mean60'] = by_day['favs'].rolling(window = window, center = False).mean()
    return by_day


def predict(df, n_preds):
    train_df = df[['favs_mean60']].reset_index()
    train_df.columns = ['ds', 'y']
    m = Prophet()
    m.fit(train_df)
    future = m.make_future_dataframe(periods=n_preds)
    return m.predict(future)


n_pred = 120
habr_by_day = prepare_data('habr_favs.csv')
gt_by_day = prepare_data('gt_favs.csv')

pred_habr_full = predict(habr_by_day, n_pred)
pred_gt_full = predict(gt_by_day, n_pred)

habr_data = pd.merge(habr_by_day[['favs_mean60']], pred_habr_full[['ds', 'yhat']].set_index('ds'), left_index = True, right_index = True, how = 'outer')
habr_data.to_csv('habr_favs_mean_pred.csv')
habr_data = pd.merge(gt_by_day[['favs_mean60']], pred_gt_full[['ds', 'yhat']].set_index('ds'), left_index = True, right_index = True, how = 'outer')
habr_data.to_csv('gt_favs_mean_pred.csv')
