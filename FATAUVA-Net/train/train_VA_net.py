import caffe
from caffe.proto import caffe_pb2
import tempfile
from pylab import *
from layers.tools import *
import sys

sys.path.append("F:/FATAUVA-Net/layers")

train_net_path = '../model/VA_Net_train.prototxt'
test_net_path = '../model/VA_Net_test.prototxt'

classes = np.asarray([i for i in range(-10,10,2)])

train_mean_file = '../prepare_data/AFEW-VA/crop/train_data.binaryproto'
test_mean_file = '../prepare_data/AFEW-VA/crop/test_data.binaryproto'

transformer = SimpleTransformer(test_mean_file)

model_weights = '../snaps/core_net_iter_500.caffemodel'


# 定义 solver
def solver(train_net_path, test_net_path=None, base_lr=0.01):

    s = caffe_pb2.SolverParameter()
    # 指定训练和测试网络
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 50  # 每训练500次，测试一次.
        s.test_iter.append(60)  # 每次test on 100 batches.

    s.max_iter = 200  # no. of times to update the net (training iterations)

    # 定义solver类型，还可以是 "Adam" 和 "Nesterov" 等.
    s.type = "SGD"

    s.base_lr = 0.01
    s.momentum = 0.9
    s.weight_decay = 5e-4

    s.lr_policy = 'step'
    s.gamma = 0.1
    # s.power = 0.75
    s.stepsize = 100

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 10
    s.snapshot = 200
    s.snapshot_prefix = '../snaps/va_net'

    s.solver_mode = caffe_pb2.SolverParameter.CPU

    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(str(s))
        return f.name


def run_solvers(niter, solvers, disp_interval=10):
    blobs = ('loss_Val', 'acc_Val')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers} for _ in blobs)

    for it in range(niter):
        for name, s in solvers:
            s.step(1)
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy() for b in blobs)

        if it % disp_interval == 0 or it +1 == niter:
            loss_disp = ';'.join('%s: loss=%.3f, acc=%2d%%' %(n, loss[n][it], np.round(100*acc[n][it]))
                                 for n, _ in solvers)
            print('%3d) %s' % (it, loss_disp))

    # Save the learned weights
    weight_dir = tempfile.mkdtemp()
    weights = {}
    for name, s in solvers:
        filename = 'weight.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])

    return loss, acc, weights


if __name__ == '__main__':

    solver_filename = solver(train_net_path, test_net_path)
    solver = caffe.get_solver(solver_filename)

    # solver.net.forward()
    # solver.test_nets[0].forward()

    solver.net.copy_from(model_weights)
    solver.test_nets[0].share_with(solver.net)

    # # 观察 label 的值 和输入图像
    # label_batch = np.array(solver.test_nets[0].blobs['label'].data, dtype=np.float16)
    # data_batch = solver.test_nets[0].blobs['data'].data.copy()
    # print(label_batch[:10])
    # vis_square(data_batch[:4].transpose(0, 2, 3, 1))

    # for k,v in solver.test_nets[0].blobs.items():
    #     print(k, v.data.shape)

    # for k,v in solver.net.params.items():
    #     print(k, v[0].data.shape)

    niter = 20
    print('Running solvers for %d iterations...' % niter)
    solvers = [('pretrained', solver)]
    loss, acc, weights = run_solvers(niter, solvers)
    print('Done')

    # train_loss = loss['pretrained']
    # train_acc = loss['pretrained']
    # train_weights = weights['pretrained']

    # solver.step(1)

    # 观察测试集中图片的 ground truth 和 estimation
    test_net = solver.test_nets[0]

    for image_index in range(5):
        figure()
        imshow(transformer.deprocess(test_net.blobs['data'].data[image_index, ...]))

        probs_Val = test_net.blobs['probs_Val'].data[image_index, ...]
        probs_Aro = test_net.blobs['probs_Aro'].data[image_index, ...]

        gtlist = np.array(test_net.blobs['label'].data[image_index], dtype=np.float16)
        estlist = [classes[probs_Val.argmax()], classes[probs_Aro.argmax()]]

        print(gtlist, estlist)
        title('GT: {} \n EST: {}'.format(gtlist, estlist))

        axis('off')
    show()



