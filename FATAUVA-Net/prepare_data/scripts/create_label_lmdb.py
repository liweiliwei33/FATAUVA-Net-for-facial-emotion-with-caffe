import caffe
import numpy as np
import scipy.misc
import scipy.io
import os.path as osp
from PIL import Image
import lmdb     # May require 'pip install lmdb' if lmdb not found
from caffe.proto import caffe_pb2
from layers.tools import *

# N = [162770, 19867, 19962]  # Number of data instances
# N = [10000, 1000]

# # CelebA labels
# N = [162770, 19962]
# M = 10      # Number of possible labels

# labels_lmdb_paths = ['../CelebA/all_data/train_label_lmdb', '../CelebA/all_data/test_label_lmdb']
# Mat file for labels N x M
# labels_mat_files = ['../CelebA/all_data/train_labels.mat', '../CelebA/all_data/test_labels.mat']
# labels = ['train_labels', 'test_labels']

# AFEW-VA labels
N = [17650, 3468]
M = 2

labels_lmdb_paths = ['../AFEW-VA/crop/train_label_lmdb', '../AFEW-VA/crop/test_label_lmdb']
labels_mat_files = ['../AFEW-VA/crop/train_labels.mat', '../AFEW-VA/crop/test_labels.mat']
labels = ['train_labels', 'test_labels']

for i in range(len(labels_mat_files)):
    X = np.zeros((N[i], M, 1, 1), dtype=np.float16)
    y = np.zeros(N[i], dtype=np.int64)
    map_size = X.nbytes * 16
    env = lmdb.open(labels_lmdb_paths[i], map_size=map_size)

    # Read the mat file and assign to X
    mat_contents = scipy.io.loadmat(labels_mat_files[i])
    X[:,:,0,0] = mat_contents[labels[i]]

    with env.begin(write=True) as txn:
        # txn is a Transaction object
        for i in range(N[i]):
            datum = caffe_pb2.Datum()
            datum.channels = X.shape[1]
            datum.height = X.shape[2]
            datum.width = X.shape[3]
            datum.data = X[i].tostring()
            datum.label = int(y[i])

            str_id = '{:08}'.format(i)
            txn.put(str_id.encode('ascii'), datum.SerializeToString())

            # target_label = np.zeros((2,1,1))
            # target_label[...] = X[i]
            # label_data = caffe.io.array_to_datum(target_label)
            # txn.put('{:08}'.format(i).encode('ascii'), label_data.SerializeToString())

            # print the progress
            print('Done label writing for data instance = ' + str(i))
            # del target_label, label_data
    env.close()
    print('labels are done!')

# # data

# image_path = 'E:/CelebA/Img/img_celeba'
# data_lmdb_paths = ['../CelebA/train_data_lmdb', '../CelebA/test_data_lmdb']
# list_files = ['../CelebA/train_data.txt', '../CelebA/test_data.txt']

# transformer = SimpleTransformer()


# for i in range(len(data_lmdb_paths)):
#
#     in_db = lmdb.open(data_lmdb_paths[i], map_size=int(1e10))
#     img_list = [line.split(' ')[0] for line in open(list_files[i])]
#
#     with in_db.begin(write=True) as in_txn:
#         for in_idx,in_ in enumerate(img_list):
#             # Load an image
#             image_file_name = img_list[in_idx]  # Get the image index
#             im = np.asarray(Image.open(
#                 osp.join(image_path, image_file_name)))
#             im = scipy.misc.imresize(im, (227, 227))  # resize
#
#             im = transformer.preprocess(im)
#             im_dat=caffe.io.array_to_datum(im)
#             in_txn.put('{:0>10d}'.format(in_idx).encode('ascii'), im_dat.SerializeToString())
#             print('data: {} [{}/{}]'.format(in_, in_idx+1, len(img_list)))
#             del im, im_dat
#     in_db.close()
#     print('data(images) are done!')