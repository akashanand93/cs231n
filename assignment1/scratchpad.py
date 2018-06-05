# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 14:04:46 2017

@author: akash.a
"""
import numpy as np

def computeLoss(W, X, Y, regularization=0):
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        score=X[i].dot(W)
        correct_score=score[Y[i]]
        for j in range(num_classes):
            if j==Y[i]:
                continue
            margin=score[j]-correct_score+1
            if margin >=0:
                loss+=margin
                
    loss /= num_train
    loss += regularization*np.sum(W*W)
    return loss


def gradientCheck(f,W, row_index, column_index):
    step = 1e-5
    Weight = W[row_index, column_index]
    W[row_index, column_index] = Weight + step
    loss1 = f(W)
    W[row_index, column_index] = Weight - step
    loss2 = f(W)    
    gradient = (loss1-loss2)/(2*step)
    return gradient

def computeGradient(W, X, Y, regularization=0):
    dW=np.zeros(W.shape)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        score=X[i].dot(W)
        correct_score=score[Y[i]]
        for j in range(num_classes):
            if j==Y[i]:
                continue
            margin=score[j]-correct_score+1
            if margin >=0:
                dW[:,j] += X[i]
                dW[:,Y[i]] -=X[i]
                loss+=margin
                
    loss /= num_train
    loss += regularization*np.sum(W*W)
    
    dW/=num_train
    dW += regularization*2*W
    return dW

X=np.random.randint(0,255,size=(2,3))
Y = np.random.randint(0,3, size=2)
W = np.random.randn(X.shape[1], 4) * 0.0001     

print(X)
print(Y)
print(W)
        
a = computeGradient(W,X,Y, 1e4)    
f = lambda w: computeLoss(w,X,Y, 1e4)  
gradientCheck(f,W,0,0)
gradientCheck(f,W,0,1)
gradientCheck(f,W,0,2)


X.dot(W)
