# -*- coding:utf-8 -*-
from pylab import *
import caffe
from caffe import layers as L, params as P
import block_def
from layers import tools

sys.path.append("F:/FATAUVA-Net/layers")

core_net_train_path = 'Core_Net_train.prototxt'
core_net_test_path = 'Core_Net_test.prototxt'


class Core_Net(object):
    def __init__(self, num_output=10):
        self.classifier_num = num_output

    def core_net_proto(self, batch_size, train=True):
        n = caffe.NetSpec()
        if train:
            source_data = '../prepare_data/CelebA/all_data/align_train_data_lmdb'
            source_label = '../prepare_data/CelebA/all_data/train_label_lmdb'
            mu = tools.get_mu('../prepare_data/CelebA/train_data.binaryproto')
        else:
            source_data = '../prepare_data/CelebA/all_data/align_test_data_lmdb'
            source_label = '../prepare_data/CelebA/all_data/test_label_lmdb'
            mu = tools.get_mu('../prepare_data/CelebA/test_data.binaryproto')

        # if train:
        #     data_layer_params = dict(batch_size=batch_size, im_shape=[227, 227],
        #                              split='train', data_root=celebA_root)
        # else:
        #     data_layer_params = dict(batch_size=batch_size, im_shape=[227, 227],
        #                              split='test', data_root=celebA_root)

        n.data = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=1,
                        transform_param=dict(scale=1. / 255, mean_value=mu),
                        input_param=dict(shape=dict(dim=[batch_size, 3, 218, 178])))
        n.label = L.Data(source=source_label, backend=P.Data.LMDB, batch_size=batch_size, ntop=1)

        # n.data, n.label = L.Python(module='multilabel_datalayer', layer='MultilabelDataLayerSync',
        #                            ntop=2, param_str=str(data_layer_params))

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = block_def.conv_bn_scale_relu(
            n.data, ks=11, nout=256, stride=4, pad=0)

        n.res0, n.conv2_bn, n.conv2_scale, n.conv2_relu = block_def.conv_bn_scale_relu(
            n.conv1, ks=9, nout=128, stride=2, pad=0)

        n_core = 8
        n_attr = 2

        # 8 层 rpoly-2 for core layer
        for num in range(n_core):
            exec('n.conv{0}_1, n.relu{0}_1, n.conv{0}_2, n.relu{0}_2, n.conv{0}_3, n.relu{0}_3, n.res{0} = block_def.rPoly2(n.res{1})'.
                 format(str(num+1), str(num)))

        # Core Layer to 4 Attribute Layer
        exec('n.res{0}_face, n.res{0}_eye, n.res{0}_eyebrow, n.res{0}_mouth = L.Split(n.res{0}, ntop=4)'.format(str(n_core)))

        # 2 层 rpoly-2 for attribute layer -- Face Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                 'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                 format(str(num+n_core+1), str(num+n_core), 'face'))

        # 2 层 rpoly-2 for attribute layer -- Eye Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                format(str(num + n_core + 1), str(num + n_core), 'eye'))

        # 2 层 rpoly-2 for attribute layer -- Eyebrow Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                format(str(num + n_core + 1), str(num + n_core), 'eyebrow'))

        # 2 层 rpoly-2 for attribute layer -- Mouth Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                format(str(num + n_core + 1), str(num + n_core), 'mouth'))

        # 3 fc for 4 layers
        n.face_label, n.eye_label, n.eyebrow_label, n.mouth_label = L.Slice(n.label, name='slice', axis=1,
                                                                            slice_point=[4, 6, 8], ntop=4)
        attr_layers = ['face', 'eye', 'eyebrow', 'mouth']
        out = [4, 2, 2, 2]

        for num in range(4):
            exec('n.fc1_{0}, n.fc1_bn_{0}, n.fc1_drop_{0} = block_def.fc_bn_drop(n.res{1}_{0}, num_output=512, '
                 'dropout_ratio=0.5)'.format(attr_layers[num], str(n_attr + n_core)))

            exec('n.fc2_{0}, n.fc2_drop_{0} = block_def.fc_drop(n.fc1_{0}, num_output=1024, '
                 'dropout_ratio=0.5)'.format(attr_layers[num]))

            exec('n.{0}_score = block_def.fc(n.fc2_{0}, num_output={1})'.format(attr_layers[num], str(out[num])))

            exec('n.loss_{0} = L.SigmoidCrossEntropyLoss(n.{0}_score, n.{0}_label)'.format(attr_layers[num]))

        if train:
            pass
        else:
            # 多标签的accuracy和单标签是不一样的，可以单独写代码来进行统计
            # n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
            # n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),
            #                              accuracy_param=dict(top_k=5))
            accuracy_layer_params = dict(batch_size=batch_size)
            n.accuracy, n.s_accuracy = L.Python(n.label, n.face_score, n.eye_score, n.eyebrow_score, n.mouth_score,
                                  module='multi_accuracy', layer='MultilabelAccuracy',
                                  ntop=2, param_str=str(accuracy_layer_params))

        return n.to_proto()
0

if __name__ == '__main__':

    # make core net
    core_model = Core_Net(num_output=10)
    tools.save_proto(core_model.core_net_proto(batch_size=50), core_net_train_path)
    tools.save_proto(core_model.core_net_proto(batch_size=50, train=False), core_net_test_path)

