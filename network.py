# neural network imports
from lasagne import layers
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum


def get_network():
    # returns a neural netork object
    # change params here
    return NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('convolution1', layers.Conv2DLayer),
            ('maxpool1', layers.MaxPool2DLayer),
            ('convolution2', layers.Conv2DLayer),
            ('maxpool2', layers.MaxPool2DLayer),
            ('convolution3', layers.Conv2DLayer),
            ('maxpool3', layers.MaxPool2DLayer),
            ('hidden1', layers.DenseLayer),
            ('hidden2', layers.DenseLayer),
            ('output', layers.DenseLayer),
        ],
        # layer properties
        # number of input points
        input_shape=(None, 1, 96, 96),
        # convolution layer properties
        convolution1_num_filters=32,
        convolution1_filter_size=(3, 3),
        maxpool1_pool_size=(2, 2),

        convolution2_num_filters=64,
        convolution2_filter_size=(2, 2),
        maxpool2_pool_size=(2, 2),

        convolution3_num_filters=128,
        convolution3_filter_size=(2, 2),
        maxpool3_pool_size=(2, 2),

        # number of units in the hidden layer
        hidden1_num_units=500,
        hidden2_num_units=500,

        output_nonlinearity=None,
        # output target value count
        output_num_units=30,

        # optimization
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,

        max_epochs=1000,
        verbose=1,
    )
