import caffe
import numpy as np


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))


class MultilabelAccuracy(caffe.Layer):

    def setup(self, bottom, top):

        # params is a python dictionary with layer parameters.
        params = eval(self.param_str)
        self.batch_size = params['batch_size']
        check_params(params)

        top[0].reshape((1))
        top[1].reshape((10))

        print_info("MultilabelAccuracy", params)

    def forward(self, bottom, top):
        """
        Load data.
        """
        acc = 0
        s_acc = np.zeros(10)

        gts = bottom[0].data.reshape(self.batch_size, 10)
        ests = np.concatenate((bottom[1].data, bottom[2].data,
                                bottom[3].data, bottom[4].data), axis=1) > 0
        for gt, est in zip(gts, ests):  # for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)

        for i in range(10):
            gt = gts[:,i]
            est = ests[:,i]
            s_acc[i] = sum([1 for (g, e) in zip(gt, est) if g == e])

        top[0].data[...] = acc / self.batch_size
        top[1].data[...] = s_acc / self.batch_size

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


def check_params(params):
    required = ['batch_size']
    for r in required:
        assert r in list(params.keys()), 'Params must include {}'.format(r)


def print_info(name, params):
    print("{} initialized with bs: {}".format(
        name,
        params['batch_size']))
