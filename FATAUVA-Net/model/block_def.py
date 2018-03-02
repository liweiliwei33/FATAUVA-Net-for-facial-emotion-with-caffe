from caffe import layers as L, params as P

# 学习参数
weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]
frozen_param = [dict(lr_mult=0)] * 2


def fc_relu_drop(bottom, num_output=1024, dropout_ratio=0.5, learn_all=True):
    param = learned_param if learn_all else frozen_param
    fc = L.InnerProduct(bottom, num_output=num_output, param=param,
                        weight_filler=dict(type='xavier', std=0.01),
                        bias_filler=dict(type='constant', value=0.2))
    relu = L.ReLU(fc, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, relu, drop


def fc_drop(bottom, num_output=1024, dropout_ratio=0.5, learn_all=True):
    param = learned_param if learn_all else frozen_param
    fc = L.InnerProduct(bottom, num_output=num_output, param=param,
                        weight_filler=dict(type='xavier', std=0.01),
                        bias_filler=dict(type='constant', value=0.2))
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, drop


def fc(bottom, num_output=1024, learn_all=True):
    param = learned_param if learn_all else frozen_param
    fc = L.InnerProduct(bottom, num_output=num_output, param=param,
                        weight_filler=dict(type='xavier', std=0.01),
                        bias_filler=dict(type='constant', value=0.2))
    return fc


def fc_relu(bottom, nout, learn_all=True):
    param = learned_param if learn_all else frozen_param
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=dict(type='xavier', std=0.01),
                        bias_filler=dict(type='constant', value=0.2))
    return fc, L.ReLU(fc, in_place=True)


def fc_bn(bottom, nout, learn_all=True):
    param = learned_param if learn_all else frozen_param
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=dict(type='xavier', std=0.01),
                        bias_filler=dict(type='constant', value=0.2))
    return fc, L.BatchNorm(fc, use_global_stats=False, in_place=True)


def fc_bn_drop(bottom, num_output=1024, dropout_ratio=0.5, learn_all=True):
    param = learned_param if learn_all else frozen_param
    fc = L.InnerProduct(bottom, num_output=num_output, param=param,
                        weight_filler=dict(type='xavier', std=0.01),
                        bias_filler=dict(type='constant', value=0.2))
    bn = L.BatchNorm(fc, use_global_stats=False, in_place=True)
    drop = L.Dropout(fc, in_place=True,
                     dropout_param=dict(dropout_ratio=dropout_ratio))
    return fc, bn, drop


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def bn_scale_relu(bottom):
    bn = L.BatchNorm(bottom, use_global_stats=False)
    scale = L.Scale(bn, scale_param=dict(bias_term=True), in_place=True)
    relu = L.ReLU(bn, in_place=True)

    return bn, scale, relu


def conv_bn_scale_relu(bottom, ks=3, nout=64, stride=1, pad=0, learn_all=True):
    param = learned_param if learn_all else frozen_param
    conv = L.Convolution(bottom, num_output=nout, kernel_size=ks, stride=stride, pad=pad,
                         param=param,
                         weight_filler=dict(type='xavier', std=0.1),
                         bias_filler=dict(type='constant', value=0.2))
    conv_bn = L.BatchNorm(conv, use_global_stats=False, in_place=True)
    conv_scale = L.Scale(conv, scale_param=dict(bias_term=True), in_place=True)
    conv_relu = L.ReLU(conv, in_place=True)

    return conv, conv_bn, conv_scale, conv_relu


def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=dict(type='xavier', std=0.1),
                         bias_filler=dict(type='constant', value=0.2))

    return conv, L.PReLU(conv, in_place=True)


def rPoly2(bottom, learn_all=True):
    res = bottom
    param = learned_param if learn_all else frozen_param
    conv1_1, relu1_1 = conv_relu(res, 3, 128, stride=1, pad=1, param=param)
    conv1_2, relu1_2 = conv_relu(relu1_1, 1, 128, stride=1, param=param)
    conv1_3, relu1_3 = conv_relu(relu1_2, 1, 128, stride=1, param=param)
    res1 = L.Eltwise(res, relu1_1, relu1_3)
    return conv1_1, relu1_1, conv1_2, relu1_2, conv1_3, relu1_3, res1


def rPoly3(bottom, learn_all=True):
    res = bottom
    param = learned_param if learn_all else frozen_param
    conv1_1, relu1_1 = conv_relu(res, 3, 128, stride=1, pad=1, param=param)
    conv1_2, relu1_2 = conv_relu(relu1_1, 1, 128, stride=1, param=param)
    conv1_3, relu1_3 = conv_relu(relu1_2, 1, 128, stride=1, param=param)
    conv1_4, relu1_4 = conv_relu(relu1_3, 1, 128, stride=1, param=param)
    res1 = L.Eltwise(res, relu1_1, relu1_2, relu1_4)
    return conv1_1, relu1_1, conv1_2, relu1_2, conv1_3, relu1_3, conv1_4, relu1_4, res1
