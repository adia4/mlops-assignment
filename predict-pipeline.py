# -*- coding: utf-8 -*-
"""
Created on Sun May 31 20:08:18 2020

@author: aditya
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("turnover.csv")
df_dummy = pd.get_dummies(df, drop_first=True)
df_dummy = df_dummy.drop('left',axis=1)

# Adding some preprocessing
df_dummy.loc[df_dummy.time_spend_company>5,'time_spend_company'] = 5
df_dummy['satisfaction_level'] = (df_dummy['satisfaction_level']**1.5)

#Modelling
filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))
df['predictions'] = model.predict(df_dummy)
df.to_csv('predicted_turnover.csv')
