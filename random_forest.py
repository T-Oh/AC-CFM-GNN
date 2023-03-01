# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:47:37 2023

@author: tobia
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import optuna

def create_dataset(path):
    power=[]
    power_std=[]
    angle=[]
    angle_std = []
    capacity=[]
    capacity_std = []
    label=[]
    for file in os.listdir(path):
        if file.startswith('data'):
            data=torch.load(path+file)
            power.append(data['x'][:,0].mean())
            power_std.append(data['x'][:,0].std())
            angle.append(data['x'][:,1].mean())
            angle_std.append(data['x'][:,1].std())
            capacity.append(data['edge_attr'][0].mean())
            capacity_std.append(data['edge_attr'][0].std())
            label.append(data['y'])
    
    #Create Dataset in correct format
    X=[power,angle,capacity,power_std, angle_std, capacity_std]
    X=np.array(X).T
    y=np.array(label)
    return X, y

class objective(object):
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test  = X_test
        self.y_train = y_train
        self.y_test  = y_test

        
    def __call__(self, trial):
       
        n_estimators = trial.suggest_int("n_estimators", 10, 1000,log =True)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 100, log=True)
        
            
        clf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split)
        clf.fit(self.X_train, self.y_train)
        
        if train_size < 1:
            preds = clf.predict(self.X_test)
            mse = mean_squared_error(self.y_test,preds)
            r2 = r2_score(self.y_test, preds)
            print(f'Labels:\n{self.y_test}')
        else:
            preds = clf.predict(self.X_train)
            mse = mean_squared_error(self.y_train,preds,squared=False)
            r2 = r2_score(self.y_train, preds)
            print(f'Labels:\n{self.y_train}')
        
        print(f'Predictions:\n{preds}')
        print(f'MSE: {mse}')
        return r2
    
    
#config
raw_path='./raw/'
processed_path='./processed/'
train_size=1

X, y = create_dataset(processed_path)

#Split data and create objective
if train_size < 1:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size)
    objective = objective(X_train, X_test, y_train, y_test, train_size)
else:
    X_train = X
    y_train = y
    objective = objective(X_train, X_train, y_train, y_train, train_size)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
best_params = study.best_params
print('Best parameters:\n')
print(best_params)

#evaluate best params to save results
clf = RandomForestRegressor(n_estimators=best_params['n_estimators'], min_samples_split=best_params['min_samples_split'])

clf.fit(X_train, y_train)

if train_size < 1:
    preds = clf.predict(X_test)
    mse = mean_squared_error(y_test,preds)
    r2 = r2_score(y_test, preds)
    print(f'Labels:\n{y_test}')
    np.savez('RF_results.npz', mse=mse, r2=r2, labels=y_test, preds=preds)
else:
    preds = clf.predict(X_train)
    mse = mean_squared_error(y_train,preds,squared=False)
    r2 = r2_score(y_train, preds)
    print(f'Labels:\n{y_train}')
    np.savez('RF_results.npz', mse=mse, r2=r2, labels=y_train, preds=preds)

print(f'Predictions:\n{preds}')
print(f'MSE: {mse}')




