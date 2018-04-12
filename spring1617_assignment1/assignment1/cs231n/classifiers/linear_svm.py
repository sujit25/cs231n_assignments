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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)     # X[i] - 1 X D, W - D X C, scores - 1 X C
        correct_class_score = scores[y[i]]
        for j in xrange(num_classes):
            if j == y[i]:  # skip for correct class
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
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

    dW /= num_train
    dW += reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    num_train = X.shape[0]
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # N = 500, D = 3073, C = 10
    # W - 3073 X 10
    # X- 500 X 3073 training images
    # y - 500 X 1

    scores = np.dot(X, W)  # X - 4000 X 3073, W - 3073 X 10, scores - 4000X 10
    y_i_scores = scores[np.arange(num_train), y]  # y_i_scores is scores of correct classes
    # print("scores shape", scores.shape)
    # print("y_i scores shape", np.matrix(y_i_scores).shape)
    # print("scores[0]", scores[:1, :])
    # print("y[0] scores", np.matrix(y_i_scores).T[:1, :])
    # print(scores[:1, :] - np.matrix(y_i_scores).T[:1, :])
    margin = np.maximum(0, scores - np.matrix(y_i_scores).T + 1)    # calculate difference from all class scores with correct class scores
    # print(margin.shape)

    # print("margin", margin[:1,:])
    # margin = np.maximum(0, scores - scores[y] + 1)

    margin[np.arange(num_train), y] = 0  # assign 0 for all the correct classes
    # margin[y] = 0
    loss += np.sum(margin)  # calculate row wise sum for all the margin
    loss += reg * np.sum(W * W)  # add regularization term
    loss /= num_train

    binary = margin
    binary[margin > 0] = 1      # binarize matrix

    row_sum = np.sum(binary, axis=1)  # compute column wise sum of classes
    # print("row sum shape", row_sum.shape)
    binary[np.arange(num_train), y] -= row_sum.T    # assign row wise sum to each row
    # print(binary[:5, :X.shape[1]])
    dW = np.dot(X.T, binary)        # compute weight gradients
    dW += reg * W
    dW /= num_train

    return loss, dW
