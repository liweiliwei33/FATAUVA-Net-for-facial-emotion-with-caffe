# -*- coding:utf-8 -*-
from pylab import *
import caffe
from caffe import layers as L, params as P
import os
import block_def
from layers import tools

sys.path.append("F:/FATAUVA-Net/layers")

au_net_train_path = 'AU_Net_train.prototxt'
au_net_test_path = 'AU_Net_test.prototxt'


class AU_Net(object):
    def __init__(self, num_output=10):
        self.classifier_num = num_output

    def au_net_proto(self, batch_size, train=True):
        n = caffe.NetSpec()
        if train:
            source_data = '../prepare_data/train_data_lmdb'
            source_label = '../prepare_data/train_label_lmdb'
        else:
            source_data = '../prepare_data/test_data_lmdb'
            source_label = '../prepare_data/test_label_lmdb'

        n.data = L.Data(source=source_data, backend=P.Data.LMDB, batch_size=batch_size, ntop=1,
                                 transform_param=dict(scale=1. / 255),
                                 input_param=dict(shape=dict(dim=[batch_size, 3, 227, 227])))
        n.label = L.Data(source=source_label, backend=P.Data.LMDB, batch_size=batch_size, ntop=1)

        n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = block_def.conv_bn_scale_relu(
            n.data, ks=11, nout=256, stride=4, pad=0)

        n.res0, n.conv2_bn, n.conv2_scale, n.conv2_relu = block_def.conv_bn_scale_relu(
            n.conv1, ks=11, nout=128, stride=4, pad=0)

        # 8 层 rpoly-2 for core layer
        for num in range(8):
            exec('n.conv{0}_1, n.relu{0}_1, n.conv{0}_2, n.relu{0}_2, n.conv{0}_3, n.relu{0}_3, n.res{0} = block_def.rPoly2(n.res{1})'.
                 format(str(num+1), str(num)))

        # Core Layer to 4 Attribute Layer
        n.res8_face, n.res8_eye, n.res8_eyebrow, n.res8_mouth = L.Split(n.res8, ntop=4)

        # 2 层 rpoly-2 for attribute layer -- Face Layer
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                 'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                 format(str(num+8+1), str(num+8), 'face'))

        # 2 层 rpoly-2 for attribute layer -- Eye Layer
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                format(str(num + 8 + 1), str(num + 8), 'eye'))

        # 2 层 rpoly-2 for attribute layer -- Eyebrow Layer
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                format(str(num + 8 + 1), str(num + 8), 'eyebrow'))

        # 2 层 rpoly-2 for attribute layer -- Mouth Layer
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_3_{2}, '
                'n.res{0}_{2} = block_def.rPoly2(n.res{1}_{2})'.
                format(str(num + 8 + 1), str(num + 8), 'mouth'))

        # Eye Layer to 2 AU Layer
        n.res10_AU6_7, n.res10_AU45 = L.Split(n.res10_eye, ntop=2)

        # 2 层 rpoly-3 for AU layer -- AU6_7
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'AU6_7'))

        # 2 层 rpoly-3 for AU layer -- AU45
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'AU45'))

        # Eyebrow Layer to 3 AU Layer
        n.res10_AU1, n.res10_AU2, n.res10_AU4 = L.Split(n.res10_eyebrow, ntop=3)

        # 2 层 rpoly-3 for AU layer -- AU1
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'AU1'))

        # 2 层 rpoly-3 for AU layer -- AU2
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'AU2'))

        # 2 层 rpoly-3 for AU layer -- AU4
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'AU4'))

        # Mouth Layer to 3 AU Layer
        n.res10_Chin, n.res10_Lip, n.res10_Mouth_AU = L.Split(n.res10_mouth, ntop=3)

        # 2 层 rpoly-3 for AU layer -- Chin
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'Chin'))

        # 2 层 rpoly-3 for AU layer -- Lip_c & Lip_u
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'Lip'))

        # 2 层 rpoly-3 for AU layer -- Mouth_AU
        for num in range(2):
            exec('n.conv{0}_1_{2}, n.relu{0}_1_{2}, n.conv{0}_2_{2}, n.relu{0}_2_{2}, n.conv{0}_3_{2}, n.relu{0}_{2},' \
                   'n.conv{0}_4_{2}, n.relu{0}_4_{2}, n.res{0}_{2}= block_def.rPoly3(n.res{1}_{2})'
                 .format(str(num+10+1),str(num+10),'Mouth_AU'))

        return n.to_proto()


if __name__ == '__main__':

    # make au net
    au_model = AU_Net(num_output=10)
    tools.save_proto(au_model.au_net_proto(batch_size=64), au_net_train_path)
    tools.save_proto(au_model.au_net_proto(batch_size=64, train=False), au_net_test_path)


