#!/usr/bin/env python
"""
Experimentos hechos con la ConvNet.  Entrenamiento con digitos de MNIST
y distorciones.  Testing con digitos de telegramas electorales argentinos de Octubre 2013.
"""

from __future__ import print_function

import sys
import os
import time
import random
import numpy as np
import theano
import theano.tensor as T

import lasagne
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset(training_set, training_labels):
    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_experimental_images(filename):
        f = open(filename, "rb")
        data = np.frombuffer(f.read(), np.uint8)
        data = data.reshape(-1, 1, 24, 24)
        return data / np.float32(255)

    def load_experimental_labels(filename):
        with open(filename) as f:
            data = f.read().splitlines()
        data=np.array(data, dtype=np.uint8)
        return data

    X_train = load_experimental_images(training_set)
    y_train = load_experimental_labels(training_labels)

    X_test = load_experimental_images('tel_images.csv')
    y_test = load_experimental_labels('tel_labels.csv')

    X_test = load_experimental_images('tes1_images.csv')
    y_test = load_experimental_labels('tes1_labels.csv')
    
    X_val = load_experimental_images('tes1_images.csv')
    y_val = load_experimental_labels('tes1_labels.csv')

    return X_train, y_train, X_val, y_val, X_test, y_test

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_cnn(input_var=None):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 24, 24), input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=6, pad=2, stride=1, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
           
            #~ nonlinearity=lasagne.nonlinearities.sigmoid,
            #~ W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details lasagne.nonlinearities.rectify
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=16, pad=2, stride=1, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)

    network = lasagne.layers.MaxPool2DLayer(network, stride=2, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    #~ network = lasagne.layers.DenseLayer(
            #~ lasagne.layers.dropout(network, p=.5),
            #~ num_units=256,
            #~ nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(batch_size=100,num_epochs=8):
    # Load the dataset
    # EXP1: sin cambios
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('exp1_images.csv', 'exp1_labels.csv')
    #~ X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('exp2_images.csv', 'exp2_labels.csv')
    # No se usa, seria cuchion barrel
    #~ X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('exp3_images.csv', 'exp3_labels.csv')
    # EXP3: Sine distortion
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('exp4_images.csv', 'exp4_labels.csv')
    #~ X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('exp5_images.csv', 'exp5_labels.csv')
    #X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('exp6_images.csv', 'exp6_labels.csv')
    X_train[ X_train > 0.4 ] = 1
    X_train[ X_train <= 0.4 ] = 0
    X_test[ X_test > 0.4 ] = 1
    X_test[ X_test <= 0.4 ] = 0
    X_train = (np.uint8(X_train))
    X_test = (np.uint8(X_test))
    training_samples = X_train.shape[0]
    validation_samples = X_val.shape[0]
    test_samples = X_test.shape[0]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    train_size = 60000
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.
    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.03, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    #~ train_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var), dtype=theano.config.floatX)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    # train_fn = theano.function([input_var, target_var], loss, updates=updates)
    train_fn = theano.function([input_var, target_var], [ loss, train_acc ], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    pred_fn = theano.function([input_var], test_prediction)

    print('train epoch,minibatch,loss,valid,test,elapsed');
    test_err = 0
    test_acc = 0
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        start_sample = 0
        if (training_samples > 60000):
            start_sample = random.randrange(0,(training_samples-60000))
        #~ print (start_sample+60000)
        #~ print (X_train[start_sample:(start_sample+60000)].shape)
        #~ print (X_train[0])
        start = time.time()
        for batch in iterate_minibatches(X_train[start_sample:(start_sample+60000)], y_train[start_sample:(start_sample+60000)], batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
        end = time.time()
        batch_test = iterate_minibatches(X_test, y_test, 100, shuffle=True)
        inputs, targets = batch_test.next()
        test_err, test_acc = val_fn(inputs, targets)
        batch_valid = iterate_minibatches(X_val, y_val, 100, shuffle=True)
        inputs, targets = batch_valid.next()
        valid_err, valid_acc = val_fn(inputs, targets)

        #~ print('train epoch %02d, minibatch %02d/%i, loss %.4f,valid: %06.2f, test: %06.2f %%,elapsed: %.4f' %
        #~ (epoch, train_batches*batch_size, train_size, train_err/train_batches, valid_acc*100, test_acc*100, (end - start)))
        print('%02d,%02d/%i,%.4f,%06.4f,%06.4f,%.4f' %
        (epoch, train_batches*batch_size, train_size, train_err/train_batches, valid_acc, test_acc, (end - start)))
    np.savez('model_mnist.npz', *lasagne.layers.get_all_param_values(network))
    # After training, we compute and print the test error:
    #~ test_batches = 0
    #~ batch_valid = iterate_minibatches(X_val, y_val, 100, shuffle=True)
    #~ inputs, targets = batch_valid.next()
    batch_valid = iterate_minibatches(X_test, y_test, test_samples, shuffle=False)
    inputs, targets = batch_valid.next()
    #~ for batch in iterate_minibatches(X_test, y_test, test_samples, shuffle=False):
    #~ inputs, targets = batch
    err, acc = val_fn(inputs, targets)
    print('samples:%d,loss %f,acc: %06.4f' % (test_samples, err, acc))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [conv layers 1] [conv layers 2] [batch size]" % sys.argv[0])
        print()
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['batch_size'] = int(sys.argv[1])
        main(**kwargs)
