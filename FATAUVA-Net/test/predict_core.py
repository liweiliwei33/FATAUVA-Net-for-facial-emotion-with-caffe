import caffe
import caffe.io
import numpy as np
import sys
import scipy.misc
from pylab import *
from layers.tools import *


sys.path.append("F:/FATAUVA-Net/layers")
caffe.set_mode_cpu()

classes = np.asarray(['Attractive','Male','No_Beard','Young','Arched_Eyebrows',
                      'Bushy_Eyebrows','Eyeglasses','Narrow_Eyes','Mouth_Slightly_Open','Smiling'])

model_def = '../model/Core_Net_test.prototxt'
model_weights = '../snaps/core_net_iter_500.caffemodel'
test_img = 'test.jpg'
test_mean_file = '../prepare_data/CelebA/test_data.binaryproto'

net = caffe.Net(model_def,
                model_weights,
                caffe.TEST)

net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          218, 178)  # image size is 227x227

transformer = SimpleTransformer(test_mean_file)

image = caffe.io.load_image(test_img)
image = scipy.misc.imresize(image, (218, 178))
net.blobs['data'].data[...] = transformer.preprocess(image)

net.forward(start='conv1')
# net.forward()

# 观察 conv1 filters: 前 64 个
# imshow(net.params['conv4_1'][0].diff[:64, 0].reshape(8,8,3,3)
#           .transpose(0,2,1,3).reshape(8*3, 3*8), cmap='gray')
# axis('off')
# plt.show()
print(net.params['conv1'][0].data[:64, 0])

est_prob = np.concatenate((net.blobs['face_score'].data, net.blobs['eye_score'].data, net.blobs['eyebrow_score'].data,
                          net.blobs['mouth_score'].data), axis=1)[0, ...]

top_inds = est_prob.argsort()[::-1]
for i in range(len(top_inds)):
    print ("%f \t %s "%(est_prob[top_inds[i]], classes[top_inds[i]]))

plt.figure()
plt.imshow(transformer.deprocess(copy(net.blobs['data'].data[0,...])))
est_list = est_prob > 0
plt.title('EST: {}'.format(classes[np.where(est_list)]))
plt.axis('off')
plt.show()
