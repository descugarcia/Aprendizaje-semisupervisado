#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:58:00 2021

@author: alumno
"""
import numpy as np
import sklearn.utils
from sklearn.base import clone

from .ssmethod import SSMethod

def measure_error(X, y, *models):
    preds = np.array([m.predict(X) for m in models])
    agreement = (preds[0] == preds).sum(axis=0) == preds.shape[0]
    error = (preds != y).sum(axis=0) == preds.shape[0]
    return error.sum() / agreement.sum()
    
def new_samples(X, *models):
    preds = np.array([m.predict(X) for m in models])
    agreement = (preds[0] == preds).sum(axis=0) == preds.shape[0]
    return X[agreement], preds[0][agreement]

def subsample(X, y, s):
    if s >= X.shape[0]:
        return X, y
    return sklearn.utils.resample(X, y, replace=False, n_samples=s)

class TriTraining(SSMethod):
    method_name = "TriTraining"
    def __init__(self, *models, model=None):
        if model is not None:
            self.models = [clone(model) for _ in range(3)]
        else:
            self.models = list(models)
        self.e = np.array([0.5] * len(self.models))
        self.l = np.array([0] * len(self.models))

    def fit(self, X, y, U):
        for model in self.models:
            # Stratify might be necessary with class imbalance, otherwise
            # X_train may contain only samples from one class, which is bad
            X_train, y_train = sklearn.utils.resample(X, y, replace=True, stratify=y)
            model.fit(X_train, y_train)
            
        converged = False
        while not converged:
            update = np.array([False for _ in range(len(self.models))])
            L = [None for _ in range(len(self.models))]
            e = [0.5 for _ in range(len(self.models))]
            for i in range(len(self.models)):
                other_models = [
                    self.models[j] for j in range(len(self.models)) if j != i
                ]
                e[i] = measure_error(X, y, *other_models)
                # Not sure, but I think this goes here
                if e[i] >= self.e[i]:
                    continue
                if e[i] < self.e[i]:
                    L[i] = new_samples(U, *other_models)
                if self.l[i] == 0:
                    self.l[i] = np.floor(e[i] / (self.e[i] - e[i]) + 1)
                if self.l[i] < L[i][0].shape[0]:
                    if e[i] * L[i][0].shape[0] < self.e[i] * self.l[i]:
                        update[i] = True
                    elif self.l[i] > e[i] / (self.e[i] - e[i]):
                        L[i] = subsample(
                            *L[i], int(np.ceil(self.e[i] * self.l[i] / e[i] + 1))
                        )
                        update[i] = True
            for i in range(len(self.models)):
                if update[i]:
                    self.models[i] = self.models[i].fit(
                        np.concatenate([X, L[i][0]]), np.concatenate([y, L[i][1]])
                    )
                    self.e[i] = e[i]
                    self.l[i] = L[i][0].shape[0]
            converged = not np.any(update)
        return self
    
    
    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        return probas.mean(axis=0)
    
    def predict(self, X):
        predicted = np.array([model.predict(X) for model in self.models])
        classes = np.unique(predicted)[:, np.newaxis, np.newaxis]
        matching_predictions = predicted == classes
        summated = matching_predictions.sum(axis=1)
        selected = summated.argmax(axis=0)

        return classes.ravel()[selected]
