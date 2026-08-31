#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:32:03 2021

@author: alumno

Implementación de:
Multilabel graph-based classification for missing labels
Yasunobu Sumikawa y Tatsurou Miyazaki
"""
import numpy as np
import sklearn.preprocessing
from sklearn.metrics.pairwise import euclidean_distances

from .kernel import restore_labels, rbf
from .ssmethod import SSMethod


def compute_P(X, sigma=1, sigma_proportion=0.05):
    P = rbf(X, sigma=sigma, sigma_proportion=sigma_proportion)
    #P = affinity_matrix(X, sigma=sigma, sigma_proportion=sigma_proportion)
    return P / P.sum(axis=1, keepdims=True)


def compute_M(X, k=5, l=None):
    effective_k = (
            int(np.ceil(X.shape[0]) * k)
            if isinstance(k, float) 
            else k
        )
    distances = euclidean_distances(X, squared=False)
    nearest_neighbors = np.argsort(distances, axis=1, kind='mergesort')
    if l is not None:
        nearest_neighbors = nearest_neighbors[nearest_neighbors < l].reshape(X.shape[0], -1)
    distances.fill(0)
    np.put_along_axis(distances, nearest_neighbors[:, 1:1+effective_k], 1, axis=1)
    # El paper insiste en que M_ij es 1 si i está entre los k vecinos más
    # cercanos de j, pero suele ser a la inversa: M_ij es uno si j está entre
    # los k vecinos más cercanos de i. Lo dejo como creo yo, pero
    # no estoy seguro.
    return distances
    #return distances.T
    
class LPAC(SSMethod):
    method_name = "LPAC"
    
    def __init__(self, beta, k=2, sigma="auto", sigma_proportion=0.05, T=20,
                 top_k_labelled=False):
        self.beta = beta
        self.sigma = sigma
        self.sigma_proportion = sigma_proportion
        self.k = k
        self.T = T
        self.top_k_labelled = top_k_labelled
        
    def fit(self, X, y, U=None):
        self.X = X
        self.y = y
        
        self.encoder = sklearn.preprocessing.OneHotEncoder().fit(
                                                             y.reshape(-1, 1))
        self.Y = np.asarray(self.encoder.transform(y.reshape(-1, 1)).todense())
        return self

    def predict_proba(self, X, original_labels=False):
        num_unlabeled = X.shape[0]
        P = compute_P(np.concatenate([self.X, X]), sigma=self.sigma, 
                      sigma_proportion=self.sigma_proportion)
        l = self.X.shape[0] if self.top_k_labelled else None
        M = compute_M(np.concatenate([self.X, X]), k=self.k, l=l)
        Y = np.concatenate([self.Y, np.zeros([X.shape[0], self.Y.shape[1]])])
        
        for i in range(self.T):
            prop = (
                self.beta * P.dot(Y) 
                + (1 - self.beta) * np.linalg.multi_dot([M, P, Y])
            )
            Y = M.dot(prop) / self.k
            # En el paper ponen Y^l_t = MY^l_t / k pero obviamente
            # las dimensiones no coinciden.
            # A) Se refiere a MY_t y asignar solo Y^l_t del resultado
            # B) se refiere a M^lY^l_t. El problema es que esto puede ignorar
            # algunos de los vecinos seleccionados.
            # Por lo tanto asumo A)
            Y[:-num_unlabeled] = (M.dot(Y) / self.k)[:-num_unlabeled]

        sums = Y.sum(axis=1)
        zero_sums = (sums == 0)
        Y[zero_sums] = [1.0]+ [0] * (Y.shape[1] - 1)
        nans = np.isnan(Y)
        row_nans = nans.sum(axis=1) >= 1
        nans_pos = nans[row_nans].argmax(axis=1)
        Y[row_nans] = 0
        Y[row_nans, nans_pos] = 1.0
        infs = np.isinf(Y)
        row_infs = infs.sum(axis=1) >= 1
        infs_pos = infs[row_infs].argmax(axis=1)
        Y[row_infs] = 0
        Y[row_infs, infs_pos] = 1.0
        Y = Y / Y.sum(axis=1, keepdims=True)
        if original_labels:
            return Y
        return Y[-num_unlabeled:]
    
    def predict(self, X, original_labels=False):
        Y = self.predict_proba(X, original_labels)
        num_unlabeled = X.shape[0]
        new_y = restore_labels(Y, self.encoder)
        if original_labels:
            return new_y[:-num_unlabeled], new_y[-num_unlabeled:]
        return new_y
    