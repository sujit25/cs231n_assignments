import numpy as np
from random import shuffle
from past.builtins import xrange

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
  num_training_examples = X.shape[0]
  num_classes = W.shape[1]

  for training_index in range(num_training_examples):
    data_row = X[training_index]                  # f- 1 X D
    scores = np.dot(data_row, W)                  # W- D X C, scores- 1 X C  - class scores
    # scores -= np.max(scores)                    # normalize values by substracting max values from each value
    correct_score = scores[y[training_index]]  # get score for correct class of this training example
    overall_score_exp = 0                         # stores sum of probabilities for each class
    for class_index in range(num_classes):
      overall_score_exp += np.exp(scores[class_index])     # calculate score and add score for each example
    loss += (-1 * np.log( np.exp(correct_score) / overall_score_exp ) )
    # loss += -1 * correct_score + np.log(overall_score_exp)    # add to loss
  loss += reg * np.sum(W * W)     # add regularization term to overall loss

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

  scores = np.dot(X, W)  # X * W , X - N X D, W - D X C , scores - N X C
  correct_scores = scores[np.arange(len(scores)), y]   # y - N X 1, correct_scores - N X 1

  loss += np.sum(-1 * np.log(np.exp(correct_scores)/np.sum(np.exp(scores), axis=1)))
  return loss, dW

