import caffe
import caffe.io
import numpy as np
import sys
import scipy.misc
from pylab import *
from layers.tools import *


sys.path.append("F:/FATAUVA-Net/layers")
caffe.set_mode_cpu()

classes = np.arange(-10,10,2)

model_def = '../model/VA_Net_test.prototxt'
model_weights = '../snaps/va_net_iter_200.caffemodel'
test_img = 'test.jpg'
test_mean_file = '../prepare_data/AFEW-VA/crop/test_data.binaryproto'


def disp_preds(net, image, labels, k=5):
    net.blobs['data'].data[0,...] = image
    net.forward(start='conv1')

    probs_Val = net.blobs['probs_Val'].data[0]
    probs_Aro = net.blobs['probs_Aro'].data[0]

    val_inds = probs_Val.argsort()[::-1][:k]
    for i in range(len(val_inds)):
        print("Top %d Valence Value: %f \t %s " % (i, probs_Val[val_inds[i]], classes[val_inds[i]]))

    aro_inds = probs_Aro.argsort()[::-1][:k]
    for i in range(len(aro_inds)):
        print("Top %d Arousal Value: %f \t %s " % (i, probs_Val[aro_inds[i]], classes[aro_inds[i]]))

    plt.figure()
    plt.imshow(transformer.deprocess(copy(net.blobs['data'].data[0, ...])))
    est_list = [classes[val_inds[0]], classes[aro_inds[0]]]
    plt.title('EST: {}'.format(est_list))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':

    net = caffe.Net(model_def,
                    model_weights,
                    caffe.TEST)

    net.blobs['data'].reshape(50,  # batch size
                              3,  # 3-channel (BGR) images
                              170, 170)  # image size is 227x227

    transformer = SimpleTransformer(test_mean_file)

    image = caffe.io.load_image(test_img)
    image = scipy.misc.imresize(image, (170, 170))

    disp_preds(net, transformer.preprocess(image), classes)


# # 一些可视化操作
# filters = net.params['conv1'][0].data
# vis_square(filters.transpose(0, 2, 3, 1))
#
# feat = net.blobs['conv1'].data[0, :36]
# vis_square(feat)


