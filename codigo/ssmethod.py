#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 10:33:35 2021

@author: alumno
"""
import abc

class SSMethod(abc.ABC):
    '''
    Clase base para los algoritmos de aprendizaje semisupervisado
    '''
    
    @abc.abstractclassmethod
    def fit(X, y, U):
        pass
    
    @abc.abstractclassmethod
    def predict(X):
        pass
    
    @abc.abstractclassmethod
    def predict_proba(X):
        pass