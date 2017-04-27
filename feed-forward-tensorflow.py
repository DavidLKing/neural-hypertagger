# Implementation of a simple MLP network with one hidden layer. Tested on the iris data set.
# Requires: numpy, sklearn, tensorflow

from __future__ import print_function
# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
# import pandas as pd
# from sklearn import datasets
# from sklearn.cross_validation import train_test_split
# import gensim
import sys
import cPickle as pickle

# example: python3 simple_mlp_tensorflow.py vectors.bin data.feats

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
# fun trick from stackoverflow
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.15)


def init_weights(shape):
    """ Weight initialization """
    weights = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(weights)

def init_bias(shape):
    """ Bias initialization """
    bias = tf.random_normal(shape, stddev=0.1)
    return tf.Variable(bias)

def forwardprop(X, w_1, b_1):
    """
    Forward-propagation.
    IMPORTANT: yhat is not softmax since TensorFlow's softmax_cross_entropy_with_logits() does that internally.
    Notes: current set up initializes at 73.35, maxes at ~85.40, max train 92.05
        2 hidden, l2, learning rate = 0.01, beta = 0.0001
    0 hid, l2, LR = 0.0001, beta = 0.0001: inialize at 1.7, max 47.25, max train 46.6 at 188 epochs
    Same as above, with +1 hidden: init at 1.6
    """
    # h_2    = tf.nn.sigmoid(tf.matmul(h_1, w_2))  # The \sigma function
    # h_1    = tf.nn.tanh(tf.matmul(tf.nn.dropout(X, 0.5, seed=RANDOM_SEED), w_1))  # The \sigma function
    # h_1    = tf.nn.relu(tf.matmul(X, w_1))  # The \sigma function
    yhat = tf.add(tf.matmul(X, w_1), b_1)  # The \varphi function
    # h_1 = tf.add(tf.matmul(X, w_1), b_1)
    # h_2 = tf.add(tf.matmul(h_1, w_2), b_2)
    # yhat = tf.add(tf.matmul(h_2, w_3), b_3)  # The \varphi function
    # yhat = tf.add(tf.matmul(h_1, w_2), b_2)  # The \varphi function
    return yhat

def main():
    if len(sys.argv) < 7:
        sys.exit("""
Please run the program like so:
python2 python2 feed-forward-tensorflow.py data/train_X.pkl data/val_X.pkl data/train_Y.pkl data/val_Y.pkl 0.001 0.1
                """)
    train_X = pickle.load(open(sys.argv[1], 'rb'))
    test_X = pickle.load(open(sys.argv[2], 'rb'))
    train_y = pickle.load(open(sys.argv[3], 'rb'))
    test_y = pickle.load(open(sys.argv[4], 'rb'))

    # training length
    length = len(train_X)

    print("Checking shapes")
    print("train_X.shape", train_X.shape)
    print("test_X.shape", test_X.shape)
    print("train_Y.shape", train_y.shape)
    print("test_Y.shape", test_y.shape)

    # Layer's sizes
    x_size = train_X.shape[1]
    y_size = train_y.shape[1]


    # l2 regularization hyperparameter
    beta = float(sys.argv[5])

    # Learning rate:
    lr = float(sys.argv[6])

    # Symbols
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])

    # Weight initializations
    w_1 = init_weights((x_size, y_size))

    # Biases
    b_1 = np.random.normal(0, 0.1, y_size)

    # Forward propagation
    yhat    = forwardprop(X, w_1, b_1)
    predict = tf.argmax(yhat, dimension=1)

    # Backward propagation
    cost    = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                    # tf.nn.sparse_softmax_cross_entropy_with_logits(
                        yhat, y)
                # l2 regularization
                + beta*tf.nn.l2_loss(w_1)
                )
    # TODO tune this
    # updates = tf.train.AdadeltaOptimizer(0.005).minimize(cost)
    updates = tf.train.AdagradOptimizer(lr).minimize(cost)
    # updates = tf.train.AdamOptimizer(0.001).minimize(cost)
    # updates = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)

    # Run SGD
    sess = tf.Session()
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # depreciated
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(100):
        # Train with each example
        # for i in range(len(train_X)):
        for i in range(length):
            sess.run(updates, feed_dict={X: train_X[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_X, y: train_y}))
        test_accuracy  = np.mean(np.argmax(test_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: test_X, y: test_y}))

        print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%"
              % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))

    sess.close()

if __name__ == '__main__':
    main()
