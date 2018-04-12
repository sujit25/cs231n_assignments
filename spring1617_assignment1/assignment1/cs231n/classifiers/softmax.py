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
    num_features = X.shape[1]
    num_classes = W.shape[1]

    for training_index in range(num_training_examples):
        data_row = X[training_index]  # f- 1 X D
        scores = np.dot(data_row, W)  # W- D X C, scores- 1 X C  - class scores
        scores -= np.max(scores)  # normalize values by substracting max values from each value
        exp_scores = np.exp(scores)
        probabilities = exp_scores / np.sum(exp_scores)
        loss += (-1 * np.log(probabilities[y[training_index]]))  # calculate loss by considering only probability of correct class
        # loss += -1 * correct_score + np.log(overall_score_exp)    # add to loss
        # for class_index in range(num_classes):
        #     dW[:, class_index] = (probabilities[class_index] - (y[class_index] == class_index)) * X[training_index]
        #
        # for feature_index in range(num_features):
        #     for class_index in range(num_classes):
        #         if class_index == y[training_index]:
        #             dW[feature_index, class_index] = X.T[feature_index][training_index] * (probabilities[class_index] - 1)
        #         else:
        #             dW[feature_index, class_index] = X.T[feature_index][training_index] * (probabilities[class_index])

        for class_index in range(num_classes):
            dW[:, class_index] += X[training_index] * (probabilities[class_index] - (class_index == y[training_index]))
    loss += reg * np.sum(W * W)  # add regularization term to overall loss

    # to compute average loss, divide by no. of training examplesddd
    loss /= num_training_examples
    dW += reg * W
    dW /= num_training_examples
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

    scores = np.dot(X, W)  # X * W , X - N X D, W - D X C , scores - N X C
    scores -= np.max(scores, axis=1, keepdims=True)  # mean normalization
    #correct_scores = scores[np.arange(len(scores)), y]  # y - N X 1, correct_scores - N X 1
    #loss += np.sum(-1 * np.log(np.exp(correct_scores) / np.sum(np.exp(scores), axis=1)))
    exp_scores = np.exp(scores)
    probabilities = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
    loss += np.sum(-1 * np.log(probabilities[np.arange(num_train), y]))
    loss += reg * np.sum(W * W)
    loss /= num_train

    dscores = probabilities.copy()
    # subtract 1 for correct class
    dscores[np.arange(num_train), y] -=1
    dW = np.dot(X.T, dscores)
    dW += reg * W
    dW /= num_train
    return loss, dW
