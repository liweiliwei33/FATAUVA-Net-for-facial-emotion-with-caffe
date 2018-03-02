import caffe
import os.path as osp
from tools import SimpleTransformer
from random import shuffle
import scipy.misc
import numpy as np
from PIL import Image
import scipy.io
from caffe import layers as L


class MultilabelDataLayerSync(caffe.Layer):

    def setup(self, bottom, top):

        self.top_names = ['data', 'label']

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        check_params(params)

        self.batch_size = params['batch_size']

        # Create a batch loader to load the images.
        self.batch_loader = BatchLoader(params, None)

        top[0].reshape(
            self.batch_size, 3, params['im_shape'][0], params['im_shape'][1])
        top[1].reshape(self.batch_size, 10)

        print_info("MultilabelDataLayerSync", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        for itt in range(self.batch_size):
            # Use the batch loader to load the next image.
            im, multilabel = self.batch_loader.load_next_image()

            # Add directly to the caffe data layer
            top[0].data[itt, ...] = im
            top[1].data[itt, ...] = multilabel

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass


class BatchLoader(object):
    def __init__(self, params, result):
        self.result = result
        self.batch_size = params['batch_size']
        self.data_root = params['data_root']
        self.im_shape = params['im_shape']
        self.split = params['split']

        list_file = '../prepare_data/{}_data.txt'.format(self.split)
        self.indexlist = [line.split(' ')[0] for line in open(list_file)]
        # get list of image indexes.
        # Read the mat file and assign to X
        mat_contents = scipy.io.loadmat('../prepare_data/{}_labels.mat'.format(self.split))
        self.X = np.zeros(mat_contents['{}_labels'.format(self.split)].shape, dtype=np.uint8)
        self.X[:, :] = mat_contents['{}_labels'.format(self.split)]

        self._cur = 0  # current image
        # this class does some simple data-manipulations
        self.transformer = SimpleTransformer()

        print("BatchLoader initialized with {} images".format(
            len(self.indexlist)))

    def load_next_image(self):
        if self._cur == len(self.indexlist):
            self._cur = 0
            # shuffle(self.indexlist)

        # Load an image
        image_file_name= self.indexlist[self._cur]  # Get the image index
        im = np.asarray(Image.open(
            osp.join(self.data_root, 'Img/img_celeba', image_file_name)))
        im = scipy.misc.imresize(im, self.im_shape)  # resize

        # do a simple horizontal flip as data augmentation
        flip = np.random.choice(2) * 2 - 1
        im = im[:, ::flip, :]

        multilabel = np.asarray(self.X[self._cur], np.float32)

        self._cur += 1
        return self.transformer.preprocess(im), multilabel
        # return im, multilabel

def load_data_annotation(index, data_root):
    classes = ('__background__',  # always index 0
               'Attractive','Male','No_Beard','Young','Arched_Eyebrows',
               'Bushy_Eyebrows','Eyeglasses','Narrow_Eyes',
               'Mouth_Slightly_Open','Smiling')
    class_to_ind = dict(list(zip(classes, list(range(11)))))

    bbox = np.zeros(4, dtype=np.uint16)
    landmarks = np.zeros((5, 2), dtype=np.int16)
    attr = np.zeros(10, dtype=np.float32)

    return {'bbox': bbox,
            'landmarks': landmarks,
            'attr': attr}

def check_params(params):
    assert 'split' in list(params.keys(
    )), 'Params must include split (train, val, or test).'

    required = ['batch_size', 'data_root', 'im_shape']
    for r in required:
        assert r in list(params.keys()), 'Params must include {}'.format(r)


def print_info(name, params):
    print("{} initialized for split: {}, with bs: {}, im_shape: {}.".format(
        name,
        params['split'],
        params['batch_size'],
        params['im_shape']))

if __name__ == '__main__':
    data_layer_params = dict(batch_size=64, im_shape=[227, 227],
                             split='train', data_root='E:/CelebA')
    data, label = L.Python(module='multilabel_datalayers', layer='multilabellayer',
                               ntop=2, param_str=str(data_layer_params))
