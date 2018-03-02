import scipy.io
import numpy as np
import matplotlib.pyplot as plt

train_label_file = '../AFEW-VA/crop/train_labels.mat'
test_label_file = '../AFEW-VA/crop/test_labels.mat'

mat_contents = scipy.io.loadmat(train_label_file)
X = np.zeros(mat_contents['train_labels'].shape, dtype=np.float16)
X[:, :] = mat_contents['train_labels']
plt.hist(X[:,0])

plt.show()