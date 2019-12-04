import sys
import numpy
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

import network
import cPickle as pickle


def load_data(training_file, testing_file, testing=False):
    # load the csv file into pandas data frame
    data_frame = read_csv(training_file)
    # convert image to a numpy array
    data_frame['Image'] = data_frame['Image'].apply(
        lambda img: numpy.fromstring(img, sep=' '))
    # drop rows with mising entries
    data_frame = data_frame.dropna()

    # grab the images out of the dataset
    X = numpy.vstack(data_frame['Image'].values) / 255.0
    X = X.astype(numpy.float32)

    # grab the target values
    y = data_frame[data_frame.columns[:-1]].values
    # normalize the target values from -1 to +1
    y = (y - 48) / 48.0

    # shuffle the data
    X, y = shuffle(X, y, random_state=42)
    y = y.astype(numpy.float32)

    return X, y


def main():

    if len(sys.argv) != 3:
        print "USAGE: python keypoint_detector.py"\
            " <training_file> <testing_file>"
        sys.exit(-1)

    # build the network
    net = network.get_network()

    X, y = load_data(sys.argv[1], sys.argv[2])
    X = X.reshape(-1, 1, 96, 96)
    net.fit(X, y)

    # save model
    net.save_params_to('model_params')
    net.save_weights_to('model_weights')

    with open('net.pickle', 'wb') as f:
        pickle.dump(net, f, -1)

    return

if __name__ == "__main__":
    main()
