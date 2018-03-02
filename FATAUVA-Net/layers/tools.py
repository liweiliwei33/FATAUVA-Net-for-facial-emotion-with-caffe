import numpy as np
from caffe.proto import caffe_pb2
import caffe
import os
import matplotlib.pyplot as plt


def get_mu(mean_file):
    # load the mean for subtraction
    mean_blob = caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(mean_file, 'rb').read())
    mean_npy = caffe.io.blobproto_to_array(mean_blob)
    mu = mean_npy[0, :, 0, 0]
    mu = np.ndarray.tolist(mu)
    return mu


def save_proto(proto, prototxt):
    with open(prototxt, 'w') as f:
        f.write(str(proto))

    os.system('python D:/caffe/python/draw_net.py %s %s.png' % (prototxt, prototxt))


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean_file=''):
        if mean_file == '':
            self.mean = np.array([128,128,128], dtype=np.float32)
        else:
            self.mean = get_mu(mean_file)
        self.scale = 1.0/255

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        im = im.transpose((2, 0, 1))

        return im

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    plt.imshow(data)
    plt.axis('off')
    plt.show()

