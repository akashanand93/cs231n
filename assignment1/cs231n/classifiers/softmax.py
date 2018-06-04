import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_feat = X.shape[1]
  num_class = W.shape[1]
  
  for i in range(num_train):
      margin = X[i].dot(W)
      loss -= margin[y[i]]      
      dW[:,y[i]] -= X[i]
      
      margin = np.exp(margin)
      loss += math.log(sum(margin))
      
      classProb = margin/sum(margin)
      dW += X[i].reshape((num_feat,1)).dot(classProb.reshape((1,num_class)))
      
  loss /=num_train
  dW /=num_train
  
  # Add regularization to the loss.
  loss += reg*np.sum(W*W)
  dW += 2*reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  
  margin = X.dot(W)
  loss -= np.sum(margin[np.arange(num_train), y])  
  
  margin = np.exp(margin)
  row_sums = np.sum(margin, axis=1)
  probability = np.divide(margin.transpose(),row_sums).transpose()

  probability[np.arange(num_train), y] -= 1
  
  loss += np.sum(np.log(row_sums))
  dW += X.transpose().dot(probability)
  
  loss /=num_train
  dW /=num_train
  
  loss += reg*np.sum(W*W)
  dW += 2*reg*W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

