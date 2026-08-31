#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 11:03:28 2021

@author: alumno

Implementation of:
Self-training semi-supervised classification based on density peaks of data
Di Wu, Mingsheng Shang, Xin Luo,Ji Xu, Huyong Yan, Weihui Deng and Guoyin Wang
"""
import sklearn.metrics
import numpy as np
import sklearn.ensemble

from .ssmethod import SSMethod

def compute_rho_delta(X, cutoff, distance="euclidean", cutoff_proportion=1.2):
    distances = sklearn.metrics.pairwise_distances(
        X, metric=distance
    )
    if cutoff == "auto":
        nearest_neighbor_distances = np.take_along_axis(
            distances, np.argsort(distances, kind="stable")[:, 1].reshape(-1, 1), axis=1
        )
        cutoff = np.median(nearest_neighbor_distances) * cutoff_proportion
    rho = (distances < cutoff).sum(axis=1)
    delta = []
    next_points = []
    for i, d in enumerate(rho):
        higher_density = rho > d
        if not np.any(higher_density):
            next_point = np.argmax(distances[i])
            next_points.append(next_point)
            delta.append(distances[i, next_point])
        else:
            nearest = np.argsort(distances[i])
            next_point = nearest[higher_density[nearest]][0]
            next_points.append(next_point)
            delta.append(distances[i, next_point])
    return rho, np.array(delta), np.array(next_points)

class DBSSC(SSMethod):
    method_name = "DBSSC"

    def __init__(self, model, cutoff="auto", distance="euclidean",
                 cutoff_proportion=1.2):
        self.model = model
        self.cutoff = cutoff
        self.cutoff_proportion = cutoff_proportion
        self.distance = distance
    
    def iteration(self, X, y, U, next_points, to_predict):
        l = X.shape[0]
        pred = self.model.predict(U[to_predict])
        mask = np.isin(np.arange(0, U.shape[0], dtype=np.int32), to_predict)
        U_subtract = U[mask]
        U = U[~mask]
        X = np.concatenate([X, U_subtract])
        y = np.concatenate([y, pred])

        # Se modifica la posición
        temp1 = next_points[l:][mask]
        temp2 = next_points[l:][~mask]
        next_points[l:l + mask.sum()] = temp1
        next_points[l + mask.sum():] = temp2
        # Se reajustan los índices de next_points
        new_pos = np.arange(0, next_points.size, dtype=np.int32)
        temp1 = new_pos[l:][mask]  # next of unlabeled instances to be added
        temp2 = new_pos[l:][~mask] # next of remaining unlabeled instances
        new_pos[l:l + mask.sum()] = temp1
        new_pos[l + mask.sum():] = temp2
        next_points = new_pos[next_points]
        self.model.fit(X, y)
        return X, y, U, next_points
    
    def fit(self, X, y, U):
        l = X.shape[0]
        rho, delta, next_points = compute_rho_delta(
            np.concatenate([X, U]), self.cutoff, self.distance, self.cutoff_proportion
        )
        self.model.fit(X, y)
        # Step 2
        while True:
            # Only pick next point from labeled instances which are from U
            to_predict = np.unique(next_points[:l])
            # transform indices from [X;U] to indices in U
            to_predict = to_predict[to_predict >= l] - l
            if to_predict.size == 0:
                # No next data of labeled points are in U
                break
            X, y, U, next_points = self.iteration(X, y, U, next_points, to_predict)
            l = X.shape[0]
        # step 3
        while True:
            to_predict = np.arange(U.shape[0])[next_points[l:] < l]
            if to_predict.size == 0:
                break
            X, y, U, next_points = self.iteration(X, y, U, next_points, to_predict)
            l = X.shape[0]
        return self
    
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
