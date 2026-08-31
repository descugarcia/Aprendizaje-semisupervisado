#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 17:36:04 2021

@author: alumno
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def rbf(X, Y=None, sigma=1, sigma_proportion=0.2):
    '''
    Calcula el resultado de un kernel rbf para cada par de elmentos de X. Si
    se pasa Y entonces se calcula para cada par de instancias (x, y).

    Se asume que cada columna corresponde a una feature y las filas
    corresponden a una instancia.

    El kernel rbf de dos vectores a y b se define como:
        k(a, b) = e ^ -((||a - b||_2)^2 / 2 * sigma)

        ||a - b||_2 es la norma euclidiana

    Argumentos
    -------------------------------
    X: numpy.ndarray shape(n, k).
        Matriz de datos 1.
    Y: numpy.ndarray shape(m, k). opcional (default=None)
        Matriz de datos 2.
    sigma: int. opcional (default=1)
        El término 'sigma'.

    Retorno
    -----------------------------------
    gram: numpy.ndarray shape(n + m, n + m).
        Gram matrix con kernel polinomial para cada par de elemntos de X. Si
        `Y` no es `None` entonces es de shape (n, m) para cada par <X, Y>
    '''
    exponent = euclidean_distances(X, Y, squared=True)
    if sigma == "auto":
        amplitude = np.median(exponent, axis=1)
        sigma = np.median(amplitude) * sigma_proportion
        sigma = np.sqrt(np.median(amplitude)/ 2)
        
    exponent /= - 2 * sigma_proportion * (sigma ** 2)
    return np.exp(exponent)

def affinity_matrix(X, sigma=1, sigma_proportion=0.1):
    W = rbf(X, sigma=sigma, sigma_proportion=sigma_proportion)
    np.fill_diagonal(W, 0)
    return W


def discretize_label_matrix(Y):
    '''
    Sets a label matrix to a discrete form in which in each row all values
    are 0 except for the maximum of that row, which will be set to 1
    '''
    Y = Y.copy()
    maximums = np.argmax(Y, axis=1)
    Y[:] = 0
    np.put_along_axis(Y, maximums.reshape(-1, 1), 1, axis=1)
    return Y

def restore_labels(Y, encoder):
    return encoder.inverse_transform(discretize_label_matrix(Y)).ravel()
