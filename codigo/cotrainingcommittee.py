#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 19:33:22 2021

@author: alumno
"""

import numpy as np
import sklearn.utils
import sklearn.ensemble
import sklearn.tree

from .ssmethod import SSMethod

class CTBC(SSMethod):
    method_name = "CTBC"
    def __init__(self, base_learner=None, ensemble_method="bagging",
                 ensemble_size=4, num_iter=50, poolsize=0.1,
                 subset_proportion=0.1):
        if base_learner is None:
            self.base_learner = (
                sklearn.tree.DecisionTreeClassifier(max_depth=1) 
                if ensemble_method == "boosting"
                else sklearn.tree.DecisionTreeClassifier()
            )
        else:
            self.base_learner = base_learner 
        self.num_iter = num_iter
        self.ensemble_size = ensemble_size
        self.poolsize = poolsize
        self.S_proportion = subset_proportion
        if ensemble_method in ["bagging", "subspaces"]:
            max_samples = 0.63 if ensemble_method == "bagging" else 1.0
            max_features = 1.0 if ensemble_method == "bagging" else 0.5
            self.ensemble_method = sklearn.ensemble.BaggingClassifier(
                estimator=self.base_learner, 
                n_estimators=self.ensemble_size,
                max_samples=max_samples,
                max_features=max_features
            )
        elif ensemble_method == "boosting":
            try:
                self.ensemble_method = sklearn.ensemble.AdaBoostClassifier(
                    estimator=self.base_learner,
                    n_estimators=self.ensemble_size,
                ).fit(np.random.random([10,2 ]), np.random.randint(0, 2, 10))
            except ValueError:
                self.ensemble_method = sklearn.ensemble.AdaBoostClassifier(
                    n_estimators=self.ensemble_size,
                )
        else:
            raise ValueError("ensemble_method must be in ['bagging', 'boosting', 'subspaces']")
    
    
    def fit(self, X, y, U):
        labels, label_prob = np.unique(y, return_counts=True)
        label_prob = label_prob / y.size
        ensemble = self.ensemble_method.fit(X, y)
        if isinstance(self.poolsize, float):
            effective_poolsize = int(np.ceil(self.poolsize * U.shape[0]))
        else:
            effective_poolsize = self.poolsize
        if effective_poolsize >= U.shape[0]:
            U_ = U
            U = np.array([])
        else:
            U_, U = sklearn.model_selection.train_test_split(U, train_size=effective_poolsize)
        total_num = U_.shape[0]
        for i in range(self.num_iter):
            if not U_.shape[0]:
                break
            confidence = ensemble.predict_proba(U_).max(axis=1)
            preds = ensemble.predict(U_)
            sorted_confidence = confidence.argsort(kind="stable")[::-1]
            indices = []
            mask = np.zeros(U_.shape[0], dtype=np.bool_)
            ind = np.arange(0, U_.shape[0], dtype=np.int32)
            for i, label in enumerate(labels):
                label_mask = preds == label
                sorted_confidence_label = sorted_confidence[
                    np.isin(sorted_confidence, ind[label_mask])
                ]
                if isinstance(self.S_proportion, int):
                    num = int(np.ceil(self.S_proportion *  label_prob[i]))
                elif isinstance(self.S_proportion, float):
                    num = int(np.ceil(self.S_proportion * total_num * label_prob[i]))
                indices.append(sorted_confidence_label[:num])
                #indices.append(sorted_confidence[label_mask][:num])
            mask[np.concatenate(indices)] = True
            S = U_[mask]
            Sy = preds[mask]
            U_ = U_[~mask]
            X = np.concatenate([X, S])
            y = np.concatenate([y, Sy])
            if S.shape[0] >= U.shape[0]:
                if not U.shape[0]:
                    pass
                else:
                    U_ = np.concatenate([U_, U]) if U_.shape[0] else U
                    U = np.array([])
            else:
                to_add, U = sklearn.model_selection.train_test_split(
                    U, train_size=S.shape[0]
                )
                U_ = np.concatenate([U_, to_add]) if U_.shape[0] else to_add
            ensemble = self.ensemble_method.fit(X, y)
        return self
    
    def predict(self, X):
        return self.ensemble_method.predict(X)
    
    def predict_proba(self, X):
        return self.ensemble_method.predict_proba(X)
