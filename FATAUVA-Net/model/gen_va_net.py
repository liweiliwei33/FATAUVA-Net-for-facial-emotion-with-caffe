# -*- coding:utf-8 -*-
from pylab import *
import caffe
from caffe import layers as L, params as P
from caffe.proto import caffe_pb2
import os
import block_def
from layers import tools

sys.path.append("F:/FATAUVA-Net/layers")

va_net_train_path = 'VA_Net_train.prototxt'
va_net_test_path = 'VA_Net_test.prototxt'

data_root = 'E:/crop_data/AFEW-VA/'
test_mean_file = '../prepare_data/AFEW-VA/crop/test_data.binaryproto'
train_mean_file = '../prepare_data/AFEW-VA/crop/train_data.binaryproto'

class VA_Net(object):
    def __init__(self, learn_all=False):
        self.learn_all = learn_all

    def va_net_proto(self, batch_size, train=True):
        n = caffe.NetSpec()
        # if train:
        #     source_data = '../prepare_data/AFEW-VA/crop/train_data_lmdb'
        #     source_label = '../prepare_data/AFEW-VA/crop/train_label_lmdb'
        #     mu = tools.get_mu('../prepare_data/AFEW-VA/crop/train_data.binaryproto')
        # else:
        #     source_data = '../prepare_data/AFEW-VA/crop/test_data_lmdb'
        #     source_label = '../prepare_data/AFEW-VA/crop/test_label_lmdb'
        #     mu = tools.get_mu('../prepare_data/AFEW-VA/crop/test_data.binaryproto')
        #
        # n.data = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=1,
        #                          transform_param=dict(scale=1. / 255, mean_value=mu),
        #                          input_param=dict(shape=dict(dim=[batch_size, 3, 170, 170])))
        # n.label = L.Data(source=source_label, backend=P.Data.LMDB, batch_size=batch_size, ntop=1)

        if train:
            data_layer_params = dict(batch_size=batch_size, im_shape=[170, 170],
                                     split='train', data_root=data_root, mean_file=train_mean_file)
        else:
            data_layer_params = dict(batch_size=batch_size, im_shape=[170, 170],
                                     split='test', data_root=data_root, mean_file=test_mean_file)

        n.data, n.label = L.Python(module='va_datalayer', layer='VADataLayerSync',
                                   ntop=2, param_str=str(data_layer_params))

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = block_def.conv_bn_scale_relu(
            n.data, ks=11, nout=256, stride=4, pad=0, learn_all=self.learn_all)

        n.res0, n.conv2_bn, n.conv2_scale, n.conv2_relu = block_def.conv_bn_scale_relu(
            n.conv1, ks=9, nout=128, stride=2, pad=0, learn_all=self.learn_all)

        n_core = 8
        n_attr = 2
        n_au = 2

        # 8 层 rpoly-2 for core layer
        for num in range(n_core):
            exec('n.conv{0}_1, n.relu{0}_1, n.conv{0}_2, n.relu{0}_2, n.conv{0}_3, n.relu{0}_3, n.res{0} ='
                 'block_def.rPoly2(n.res{1}, learn_all=self.learn_all)'.format(str(num + 1), str(num)))

        # Core Layer to 4 Attribute Layer
        # exec('n.res{0}_face, n.res{0}_eye, n.res{0}_eyebrow, n.res{0}_mouth = L.Split(n.res{0}, ntop=4)'
        #      .format(str(n_core)))
        exec('n.res{0}_eye, n.res{0}_eyebrow, n.res{0}_mouth = L.Split(n.res{0}, ntop=3)'
             .format(str(n_core)))

        # # 2 层 rpoly-2 for attribute layer -- Face Layer
        # for num in range(n_attr):
        #     exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
        #         'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2}, learn_all=self.learn_all)'.
        #         format(str(num + n_core + 1), str(num + n_core), 'face'))

        # 2 层 rpoly-2 for attribute layer -- Eye Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2}, learn_all=self.learn_all)'.
                format(str(num + n_core + 1), str(num + n_core), 'eye'))

        # 2 层 rpoly-2 for attribute layer -- Eyebrow Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2}, learn_all=self.learn_all)'.
                format(str(num + n_core + 1), str(num + n_core), 'eyebrow'))

        # 2 层 rpoly-2 for attribute layer -- Mouth Layer
        for num in range(n_attr):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2}, learn_all=self.learn_all)'.
                format(str(num + n_core + 1), str(num + n_core), 'mouth'))

        ########################################
        # Eye Layer to 2 AU Layer
        exec('n.res{0}_AU6_7, n.res{0}_AU45 = L.Split(n.res{0}_eye, ntop=2)'.format(str(n_core + n_attr)))

        # 2 层 rpoly-3 for AU layer -- AU6_7
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'AU6_7'))

        # 2 层 rpoly-3 for AU layer -- AU45
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'AU45'))

        # Eyebrow Layer to 3 AU Layer
        exec('n.res{0}_AU1, n.res{0}_AU2, n.res{0}_AU4 = L.Split(n.res{0}_eyebrow, ntop=3)'.format(str(n_core+n_attr)))

        # 2 层 rpoly-3 for AU layer -- AU1
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'AU1'))

        # 2 层 rpoly-3 for AU layer -- AU2
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'AU2'))

        # 2 层 rpoly-3 for AU layer -- AU4
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'AU4'))

        # Mouth Layer to 3 AU Layer
        exec('n.res{0}_Chin, n.res{0}_Lip, n.res{0}_Mouth_AU = L.Split(n.res{0}_mouth, ntop=3)'.format(str(n_core+n_attr)))

        # 2 层 rpoly-3 for AU layer -- Chin
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'Chin'))

        # 2 层 rpoly-3 for AU layer -- Lip_c & Lip_u
        # for num in range(n_au):
        #     exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
        #            'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
        #          .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'Lip'))

        exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
             'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
             .format(str(n_core + n_attr + 1), str(n_core + n_attr), 'Lip'))

        # exec('n.res{0}_Lip_c, n.res{0}_Lip_u = L.Split(n.res{0}_Lip, ntop=2)'.format(str(n_core + n_attr + 1)))
        exec('n.res{0}_Lip_c = L.Split(n.res{0}_Lip, ntop=1)'.format(str(n_core + n_attr + 1)))

        exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
             'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
             .format(str(n_core + n_attr + 2), str(n_core + n_attr +1), 'Lip_c'))

        # exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
        #      'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
        #      .format(str(n_core + n_attr + 2), str(n_core + n_attr +1), 'Lip_u'))

        # 2 层 rpoly-3 for AU layer -- Mouth_AU
        for num in range(n_au):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+n_core+n_attr+1),str(num+n_core+n_attr),'Mouth_AU'))

        ########################################
        # AU6_7, AU45, AU4, Lip_c, Mouth_AU for Valence layer
        exec('n.res_Val = L.Concat(n.res{0}_AU6_7, n.res{0}_AU45, n.res{0}_AU4, n.res{0}_Lip_c, n.res{0}_Mouth_AU, axis=1)'\
            .format(str(n_core+n_attr+n_au)))

        # AU45, AU1, AU2, AU4, Chin for Arousal layer

        exec('n.res_Aro = L.Concat(n.res{0}_AU45, n.res{0}_AU1, n.res{0}_AU2, n.res{0}_AU4, n.res{0}_Chin, axis=1)' \
            .format(str(n_core + n_attr + n_au)))

        # va labels
        n.Val_label, n.Aro_label = L.Slice(n.label, name='slice', axis=1, slice_point=[1], ntop=2)
        va_layers = ['Val', 'Aro']
        out = [10, 10]

        for num in range(2):
            exec('n.fc1_{0}, n.fc1_bn_{0}, n.fc1_drop_{0} = block_def.fc_bn_drop(n.res_{0}, num_output=1024, '
                 'dropout_ratio=0.5)'.format(va_layers[num]))

            exec('n.fc2_{0}, n.fc2_bn_{0}, n.fc2_drop_{0} = block_def.fc_bn_drop(n.fc1_{0}, num_output=1024, '
                 'dropout_ratio=0.5)'.format(va_layers[num]))

            exec('n.{0}_score = block_def.fc(n.fc2_{0}, num_output={1})'.format(va_layers[num], str(out[num])))

            exec('n.loss_{0} = L.SoftmaxWithLoss(n.{0}_score, n.{0}_label)'.format(va_layers[num]))

            exec('n.acc_{0} = L.Accuracy(n.{0}_score, n.{0}_label)'.format(va_layers[num]))

        if train:
            pass
        else:
            n.probs_Val = L.Softmax(n.Val_score)
            n.probs_Aro = L.Softmax(n.Aro_score)

        return n.to_proto()


if __name__ == '__main__':

    # make va net
    va_model = VA_Net(learn_all=False)
    tools.save_proto(va_model.va_net_proto(batch_size=50), va_net_train_path)
    tools.save_proto(va_model.va_net_proto(batch_size=50, train=False), va_net_test_path)
