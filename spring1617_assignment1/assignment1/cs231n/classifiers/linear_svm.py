import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:     # skip for correct class
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # update partial derivative w.r.t to w_j
        dW[:, j] += X[i]

        # loss when computed with w.r.t to w_y
        dW[:, y[i]] += -1 * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # N = 500, D = 3073, C = 10
  # W - 3073 X 10
  # X- 500 X 3073 training images
  # y - 500 X 1

  scores = np.dot(X, W)         # X - 4000X3073, W - 3073 X 10, scores - 4000X 10
  y_i_scores = scores[np.arange(len(scores)), y]    # y_i_scores is scores of correct classes
  margin = np.maximum(0, scores - np.matrix(y_i_scores).T + 1)   # calculate difference from all class scores with correct class scores
  margin[np.arange(len(scores)), y] = 0     # assign 0 for all the correct classes
  loss += np.sum(margin, axis = 0)     # calculate row wise sum for all the margin
  loss += reg * np.sum(W * W)       # add regularization term
  print(np.shape(loss))
  # correct_class_score = scores[y]  # - 4000 X 1 - stores correct class score for each image
  # margin = (scores + 1) - scores[y] # 4000 X 10
  # margin[y] = 0
  # for index in range(len(margin)):
  #     for marginal_score_index in range(len(margin[index])):
  #       if margin[index][marginal_score_index] > 0 :
  #         loss += margin[index][marginal_score_index]

  binary = margin









  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
