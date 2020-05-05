# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:23:46 2018

@author: deept
"""
#imports
import numpy as np
import pandas as pd
from patsy import dmatrices
import warnings

def sigmoid(x):
    return 1/(1+np.exp(-x))

np.random.seed(0)
#step 1: Defining model parameters or Hyperparameters:
tol=1e-8 #Convergence Tolerance
lam=None #l2-regularization
max_iter=20


#Data creation
r=0.95 #Covariance b/w x&z
n=1000 #dataset size
sigma=1 #noise variance

#Model settings
beta_x, beta_z, beta_v = -4, 0.9, 1
var_x, var_z, var_v = 1, 1, 4

#Formulele
formula = 'y ~ x + z + v + np.exp(x) + I(v**2 + z)'

#step 2: Generating and organizing data
x, z = np.random.multivariate_normal([0.0], [[var_x,r],[r,var_z]], n).T
v = np.random.normal(0,var_v,n)**3

#Pandas dataframe time:
A = pd.DataFrame({'x' : x, 'z' : z, 'v' : v})
#Computing log odds and adding it to the dataframe:
A['log_odds'] = sigmoid(A[['x','z','v']].dot([beta_x,beta_z,beta_v]) + sigma*np.random.normal(0,1,n))

#Computing probablility distribution:
A['y'] = [np.random.binomial(1,p) for p in A.log_odds]

#creating a dataframe that has the input data, formula and outputs
y, X = dmatrices(formula, A, return_type='dataframe')
 