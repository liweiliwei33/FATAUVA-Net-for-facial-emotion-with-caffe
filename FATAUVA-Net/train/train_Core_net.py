# -*- coding:utf-8 -*-

from pylab import *
from layers.tools import *

sys.path.append("F:/FATAUVA-Net/layers")

# 网络定义路径
core_net_train_path = '../model/Core_Net_train.prototxt'
core_net_test_path = '../model/Core_Net_test.prototxt'
solver_config_path = 'Core_Net_solver.prototxt'

# 数据集路径及类别
celebA_root = 'E:/CelebA'
classes = np.asarray(['Attractive','Male','No_Beard','Young','Arched_Eyebrows',
                      'Bushy_Eyebrows','Eyeglasses','Narrow_Eyes','Mouth_Slightly_Open','Smiling'])

train_mean_file = '../prepare_data/CelebA/train_data.binaryproto'
test_mean_file = '../prepare_data/CelebA/test_data.binaryproto'


transformer = SimpleTransformer(test_mean_file)
# transformer.set_mean(mu)          # subtract the dataset-mean value in each channel

# 定义 solver
def core_solver():

    s = caffe_pb2.SolverParameter()
    # 指定训练和测试网络
    s.train_net = core_net_train_path
    s.test_net.append(core_net_test_path)
    s.test_interval = 100  # 每训练100次，测试一次.
    s.test_iter.append(100)  # 每次test on 100 batches.
    s.max_iter = 10000  # no. of times to update the net (training iterations)

    # 定义solver类型: SGD, Adam, AdaDelta, AdaGrad, Adam, Nesterov, RMSProp
    s.type = "Adam"

    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4

    s.lr_policy = 'inv'
    s.gamma = 0.0001
    s.power = 0.75

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 10
    s.snapshot = 500
    s.snapshot_prefix = '../snaps/core_net'

    s.solver_mode = caffe_pb2.SolverParameter.CPU

    return s


def hamming_distance(gt, est):
    return sum([1 for (g, e) in zip(gt, est) if g == e]) / float(len(gt))


def check_accuracy(net, num_batches, batch_size=50):
    acc = 0.0
    s_acc = np.zeros(10)
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data.reshape(batch_size, 10)
        ests = np.concatenate((net.blobs['face_score'].data, net.blobs['eye_score'].data,
                        net.blobs['eyebrow_score'].data, net.blobs['mouth_score'].data), axis=1) > 0
        for gt, est in zip(gts, ests):  # for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)

        for i in range(10):
            gt = gts[:,i]
            est = ests[:,i]
            s_acc[i] += sum([1 for (g, e) in zip(gt, est) if g == e])

    return acc / (num_batches * batch_size), s_acc / (num_batches * batch_size)


def check_base_accuracy(net, num_batches, batch_size=50):
    acc = 0.0
    for t in range(num_batches):
        net.forward()
        gts = net.blobs['label'].data.reshape(batch_size, 10)
        ests = np.zeros((batch_size, 10))
        for gt, est in zip(gts, ests):  # for each ground truth and estimated label vector
            acc += hamming_distance(gt, est)
    return acc / (num_batches * batch_size)


if __name__ == '__main__':

    with open(solver_config_path, 'w') as f:
        f.write(str(core_solver()))

    solver = None
    solver = caffe.get_solver(solver_config_path)

    solver.net.forward()
    solver.test_nets[0].forward()
    # solver.net.copy_from('../snaps/.caffemodel')
    # solver.test_nets[0].share_with(solver.net)
    # solver.step(1)

    # # 观察网络结构
    # for k, v in solver.net.blobs.items():
    #     print(k,v.data.shape)
    #
    # for k, v in solver.net.params.items():
    #     print(k,v[0].data.shape)

    # # 自定义训练，绘制 loss 和 Accuracy 曲线
    # niter = 10000
    # test_interval = int(niter / 100)
    # train_loss = zeros(niter)
    # test_acc = zeros(int(np.ceil(niter / test_interval)))
    # # output = zeros((niter, 8, 4))
    #
    # # 单标签准确率
    # attr_acc = zeros((int(np.ceil(niter / test_interval)), 10), dtype=np.float32)
    #
    # for it in range(niter):
    #     solver.step(1)
    #
    #     train_loss[it] = (solver.net.blobs['loss_face'].data + solver.net.blobs['loss_eye'].data +
    #                       solver.net.blobs['loss_eyebrow'].data + solver.net.blobs['loss_mouth'].data) / 4
    #
    #     # store the output on the first test batch
    #     # (start the forward pass at conv1 to avoid loading new data)
    #     # solver.test_nets[0].forward(start='conv1')
    #     # output[it] = solver.test_nets[0].blobs['face_score'].data[:8]
    #
    #     if it % test_interval == 0:
    #         print('Iteration', it, 'testing...')
    #         test_acc[it // test_interval], attr_acc[it // test_interval, :] = \
    #             check_accuracy(solver.test_nets[0], 100, 50)
    #
    # _, ax1 = subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(arange(niter), train_loss)
    # ax2.plot(test_interval * arange(len(test_acc)), test_acc, 'r')
    # ax1.set_xlabel('iteration')
    # ax1.set_ylabel('train loss')
    # ax2.set_ylabel('test accuracy')
    # ax2.set_title('Custom Test Accuracy: {:.2f}'.format(test_acc[-1]))

    # # 观察测试集中图片的 ground truth 和 estimation
    # test_net = solver.test_nets[0]
    # # print('Baseline accuracy:{0:.4f}'.format(check_base_accuracy(test_net, 1000//50, 50)))
    #
    # for image_index in range(5):
    #     figure()
    #     imshow(transformer.deprocess(test_net.blobs['data'].data[image_index, ...]))
    #     gtlist = test_net.blobs['label'].data[image_index, ...].astype(np.int).reshape(10)
    #     estlist = np.concatenate((test_net.blobs['face_score'].data, test_net.blobs['eye_score'].data,
    #                                 test_net.blobs['eyebrow_score'].data, test_net.blobs['mouth_score'].data), axis=1)[image_index, ...] > 0
    #
    #     # estlist = test_net.blobs['classifier'].data[image_index, ...] > 0
    #     # print(test_net.blobs['classifier'].data[image_index, ...])
    #     print(np.concatenate((test_net.blobs['face_score'].data, test_net.blobs['eye_score'].data,
    #                           test_net.blobs['eyebrow_score'].data, test_net.blobs['mouth_score'].data), axis=1)[image_index, ...])
    #     # print(gtlist, estlist)
    #     title('GT: {} \n EST: {}'.format(classes[np.where(gtlist)], classes[np.where(estlist)]))
    #     # plt.title('EST: {}'.format(classes[np.where(estlist)]))
    #     axis('off')
    # show()

    # 保存网络
    # test_net.save('../snaps/CoreNet.caffemodel')

    # MNIST 观察输入图像
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(solver.net.blobs['data'].data[:8, 0].transpose(1, 0, 2).reshape(28, 8 * 28), cmap='gray')
    # axis('off')

    # 观察 conv1 filters: 前 64 个
    # ax = fig.add_subplot(211)
    # ax.imshow(solver.net.params['conv1'][0].diff[:64, 0].reshape(8,8,3,3)
    #           .transpose(0,2,1,3).reshape(8*3, 3*8), cmap='gray')
    # axis('off')
    # plt.show()

    # MNIST 观察 prediction scores
    # for i in range(4):
    #     figure(figsize=(2, 2))
    #     imshow(solver.test_nets[0].blobs['data'].data[i, 0], cmap='gray')
    #     figure(figsize=(10, 2))
    #     imshow(exp(output[:50, i].T) / exp(output[:50, i].T).sum(0), interpolation='nearest', cmap='gray')
    #     xlabel('iteration')
    #     ylabel('label')
