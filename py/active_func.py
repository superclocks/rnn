import numpy as np

def Sigmoid(x):
    return 1/(1 + np.exp(-x))   
def DerSigmoid(x):
    return Sigmoid(x) * (1.0 - Sigmoid(x))
    
def Tanh(x): 
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
def DerTanh(x):
    return 1.0 - Tanh(x) * Tanh(x)

def Softmax(x):
    r = np.exp(x)
    return  r / sum(r)
def DerSoftmax(x):
    return Softmax(x) * (1.0 - Softmax(x)) 