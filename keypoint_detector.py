import sys
import numpy
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import network
import cPickle as pickle
from matplotlib import pyplot


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


def load_data(training_file, testing_file, testing=False):
    # load the csv file into pandas data frame
    data_frame = read_csv(training_file)
    # convert image to a numpy array
    data_frame['Image'] = data_frame['Image'].apply(
        lambda img: numpy.fromstring(img, sep=' '))
    # drop rows with mising entries
    # data_frame = data_frame.dropna()

    # grab the images out of the dataset
    X = numpy.vstack(data_frame['Image'].values) / 255.0
    X = X.astype(numpy.float32)

    # grab the target values
    # y = data_frame[data_frame.columns[:-1]].values
    # normalize the target values from -1 to +1
    # y = (y - 48) / 48.0

    # shuffle the data
    # X, y = shuffle(X, y, random_state=42)
    # y = y.astype(numpy.float32)

    return X, None


def normalize(value):
    return str((value * 48.0) + 48.0)


def main():
    if len(sys.argv) != 5:
        print "USAGE python keypoint_detector.py <model>" \
            " <params_file> <weights_file> <training_file>"
        sys.exit(-1)

    net = network.get_network()
    # net.load_params_from(sys.argv[2])
    net.load_weights_from(sys.argv[3])
    net.initialize()

    X, y = load_data(sys.argv[4], '', testing=True)
    X = X.reshape(-1, 1, 96, 96)
    preds = net.predict(X)
    for pred in preds:
        print ','.join(map(normalize, pred))
    return


if __name__ == "__main__":
    main()
