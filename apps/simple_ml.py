import needle as ndl
import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')


def parse_mnist(image_filesname, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    # BEGIN YOUR SOLUTION
    byte_order = 'big'
    X = []
    with gzip.open(image_filesname) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2051:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_rows = int.from_bytes(f.read(4), byteorder=byte_order)
        num_of_cols = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            img_one_dim = np.frombuffer(
                f.read(num_of_rows * num_of_cols), dtype=np.uint8).astype(np.float32)
            X.append(img_one_dim)
    X = np.array(X)
    min = np.min(X)
    max = np.max(X)
    X = (X - min) / (max - min)
    y = []
    with gzip.open(label_filename) as f:
        magic_num = int.from_bytes(f.read(4), byteorder=byte_order)
        if magic_num != 2049:
            raise Exception('decode error!')
        num_of_imgs = int.from_bytes(f.read(4), byteorder=byte_order)
        for i in range(num_of_imgs):
            label = np.frombuffer(f.read(1), dtype=np.uint8)
            y.append(label)

    return X, np.array(y).squeeze()

    # END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    # BEGIN YOUR SOLUTION

    exps = ndl.exp(Z)
    exps_up = ndl.summation(ndl.multiply(exps, y_one_hot), axes=1)
    #print('exps up: ',exps_up)
    exps_down = ndl.summation(exps, axes=1)
    #print('exps down: ',exps_down)
    return -1 * ndl.summation(ndl.log(exps_up / exps_down))/ndl.Tensor(Z.shape[0])
    # END YOUR SOLUTION


def _one_hot_labels(data, num_classes):
    """

    :param data: (num,) ndarray
    :return: (num, num_classes)
    """
    num_examples = len(data)
    # (num_examples, num_classes)
    one_hot_labels = np.zeros((num_examples, num_classes))
    indices = np.stack((np.arange(len(data)), data), axis=1)
    one_hot_labels[indices[:, 0], indices[:, 1]] = 1
    return one_hot_labels


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    # BEGIN YOUR SOLUTION
    start_indices = np.arange(0, len(X), batch)
    num_classes = W2.shape[1]
    for i in start_indices:
        if i + batch > len(X):
            break
        batch_X = ndl.Tensor(X[i:i + batch])  # (batch_size, input_dim)
        batch_y = y[i:i + batch]  # (batch_size,)
        labels = ndl.Tensor(_one_hot_labels(batch_y, num_classes))

        preds = ndl.matmul(ndl.relu(ndl.matmul(batch_X, W1)), W2)
        loss = softmax_loss(preds, labels)

        loss.backward(lr)

        # this would lower the performance!
        # W1 -= W1.grad
        # W2 -= W2.grad
        W1 = ndl.Tensor(W1.numpy() - W1.grad.numpy())
        W2 = ndl.Tensor(W2.numpy() - W2.grad.numpy())

    return W1, W2
    # END YOUR SOLUTION


# CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h, y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
