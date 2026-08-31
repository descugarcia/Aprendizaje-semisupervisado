#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:02:21 2021

@author: alumno

Implementación de:
Learning with Local and Global Consistency
Dengyong Zhou, Olivier Bousquet, Thomas Navin Lal, Jason Weston, y 
Bernhard Schölkopf
-----------------------------------------
Label Propagation through Linear Neighborhoods
Fei Wang and Changshui Zhang
"""
import abc

import numpy as np
import scipy as sp
from sklearn.metrics.pairwise import euclidean_distances
import sklearn.preprocessing
import cvxopt

from .ssmethod import SSMethod
from .kernel import restore_labels, affinity_matrix, rbf


def compute_W(X, num_neighbors=10):
    '''
    Computation of affinity matrix as in 
    "Label Propagation through Linear Neighborhoods". Copied from
    https://github.com/BioMedicalBigDataMiningLab/CD-LNLP/blob/3d53d3f2c3749269c1ec5d160a9da94366719a3b/LNLP_method.py#L43
    '''
    distances = euclidean_distances(X, squared=False)
    nearest_indices = np.argsort(distances, axis=1, kind='mergesort')
    W = np.zeros([X.shape[0], X.shape[0]])
    
    # For the optimization problem
    q = cvxopt.matrix(np.zeros(num_neighbors))
    G = cvxopt.matrix(np.identity(num_neighbors) * -1)
    h = cvxopt.matrix(np.zeros(num_neighbors))
    A = cvxopt.matrix(np.ones([1, num_neighbors]))
    b = cvxopt.matrix(1.0)
    for i in range(X.shape[0]):
        nearest_neighbors_indices = nearest_indices[i, 1:1+num_neighbors]
        diffs = (X[i] - X[nearest_neighbors_indices]).astype(np.float64)
        P = 2 * cvxopt.matrix(diffs.dot(diffs.T))
        try:
            solution = np.array(cvxopt.solvers.qp(P, q, G, h, A, b)["x"]).ravel()
        except ValueError:
            # No solution can be achieved
            continue
        W[i, nearest_neighbors_indices] = solution
    return W

def inverse_square_root(X):
    sqrt = sp.linalg.sqrtm(X)
    try:
        return np.linalg.inv(sqrt)
    except np.linalg.LinAlgError:
        print("linalg error in inverse square root")
        #raise
        return np.linalg.pinv(sqrt)
      
    
def compute_D(W):
    return np.diag(np.sum(W, axis=1))

def compute_S(X, sigma=1, sigma_proportion=0.2):
    W = affinity_matrix(X, sigma=sigma, sigma_proportion=sigma_proportion)
    D = compute_D(W)
    D_is = inverse_square_root(D)
    return np.linalg.multi_dot([D_is, W, D_is])


def compute_F(S, Y, alpha=0.5, beta=1):
    first_term = np.identity(S.shape[0]) - alpha * S
    try:
        inverse = np.linalg.inv(first_term)
    except np.linalg.LinAlgError:
        print("linalg error in inverse square root")
        #raise
        inverse =  np.linalg.pinv(first_term)
    return np.asarray((beta * inverse).dot(Y))


def compute_P(X, sigma=1, sigma_proportion=0.2):
    W = affinity_matrix(X, sigma=sigma, sigma_proportion=sigma_proportion)
    D = compute_D(W)
    try:
        D_inv = np.linalg.inv(D)
    except np.LinAlgError:
        D_inv = np.linalg.pinv(D)
    return D_inv.dot(W)


def compute_P_fick(X, gamma=1, sigma=1, sigma_proportion=0.2,
                   gamma_proportion=0.5):
    distances = 1 / rbf(X, sigma=sigma, sigma_proportion=sigma_proportion)
    distances = distances ** - 2
    np.fill_diagonal(distances, 0)
    max_gamma = np.max(distances.sum(axis=1)) ** -1
    if gamma == 'auto':
        gamma = max_gamma * gamma_proportion
    if gamma > max_gamma:
        raise ValueError("gamma is set to %f, but max value is %f" % (gamma, max_gamma))
    diagonal_values = 1 - gamma * (distances.sum(axis=1))
    distances *= gamma
    np.fill_diagonal(distances, diagonal_values)
    return distances

class LGMethod(SSMethod):
    '''
    Clase base para los métodos basados en LGC
    '''
    def __init__(self, alpha=0.5, mu=None):
        if mu is not None:
            self.alpha = 1 / (1 + mu)
            self.beta = mu / (1 + mu)
        else:
            self.alpha = alpha
            self.beta = 1 - alpha
        
    def fit(self, X, y, U=None):
        self.X = X
        self.y = y
        
        self.encoder = sklearn.preprocessing.OneHotEncoder().fit(
                                                             y.reshape(-1, 1))
        self.Y = self.encoder.transform(y.reshape(-1, 1)).todense()
        return self
    
    @abc.abstractmethod
    def compute_base_matrix(self, X, **kwargs):
        pass
    
    def predict_proba(self, X, original_labels=False):
        num = X.shape[0]
        Y = np.concatenate([self.Y, np.zeros([X.shape[0], self.Y.shape[1]])])
    
        base_matrix = self.compute_base_matrix(np.concatenate([self.X, X]))
        F = compute_F(base_matrix, Y, self.alpha, self.beta)
            
        # return labels
        sums = F.sum(axis=1)
        zero_sums = (sums == 0)
        F[zero_sums] = [1.0]+ [0] * (F.shape[1] - 1)
        nans = np.isnan(F)
        row_nans = nans.sum(axis=1) >= 1
        nans_pos = nans[row_nans].argmax(axis=1)
        F[row_nans] = 0
        F[row_nans, nans_pos] = 1.0
        infs = np.isinf(F)
        row_infs = infs.sum(axis=1) >= 1
        infs_pos = infs[row_infs].argmax(axis=1)
        F[row_infs] = 0
        F[row_infs, infs_pos] = 1.0
        F = F / F.sum(axis=1, keepdims=True)
        if original_labels:
            return F
        return F[-num:]
    
    def predict(self, X, original_labels=False):
        Y = self.predict_proba(X, original_labels=False)
        num = X.shape[0]
        new_y = restore_labels(Y, self.encoder)
        if original_labels:
            return new_y[:-num], new_y[-num:]
        return new_y

    

class LGC(LGMethod):
    method_name = "LGC"
    def __init__(self, sigma="auto", alpha=0.5, mu=None, sigma_proportion=0.2):
        LGMethod.__init__(self, alpha, mu)
        self.sigma = sigma
        self.sigma_proportion = sigma_proportion
        
    
    def compute_base_matrix(self, X):
        return compute_S(X, self.sigma, sigma_proportion=self.sigma_proportion)
    

class LGCVariant1(LGC):
    method_name = "LGCVariant1"

    def compute_base_matrix(self, X):
        return compute_P(X, self.sigma, sigma_proportion=self.sigma_proportion)
    
class LGCVariant2(LGC):
    method_name = "LGCVariant2"

    def compute_base_matrix(self, X):
        return compute_P(X, self.sigma, sigma_proportion=self.sigma_proportion).T
        
    
class LinearNeighborhood(LGMethod):
    method_name = "LinearNeighborhood"
    def __init__(self, alpha=0.5, mu=None, num_neighbors=10):
        LGMethod.__init__(self, alpha, mu)
        self.num_neighbors = num_neighbors
        
    def compute_base_matrix(self, X):
        '''
        Computation of affinity matrix as in 
        "Label Propagation through Linear Neighborhoods". Copied from
        https://github.com/BioMedicalBigDataMiningLab/CD-LNLP/blob/3d53d3f2c3749269c1ec5d160a9da94366719a3b/LNLP_method.py#L43
        '''
        num_neighbors = (
            int(np.ceil(X.shape[0]) * self.num_neighbors)
            if isinstance(self.num_neighbors, float) 
            else self.num_neighbors
        )
        return compute_W(X, num_neighbors)
    


class Fick(LGMethod):
    method_name = "Fick"
    def __init__(self, sigma="auto", alpha=0.5, mu=None, gamma="auto",
                 sigma_proportion=0.2, gamma_proportion=0.5):
        LGMethod.__init__(self, alpha=alpha, mu=mu)
        self.sigma = sigma
        self.sigma_proportion = sigma_proportion
        self.gamma = gamma
        self.gamma_proportion = gamma_proportion
        
    def compute_base_matrix(self, X):
        return compute_P_fick(
                X, self.gamma, self.sigma,
                sigma_proportion=self.sigma_proportion, 
                gamma_proportion=self.gamma_proportion
        )
