from convolution import convolution
from maxpool import max_pool
from cost import *
import numpy as np
import gzip


def extract_data(filename, num_images, img_w):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(img_w * img_w * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, img_w * img_w)
        return data


def extract_label(filename, num_images):
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1*num_images)
        labels = np.frombuffer(buf, dtype=np.unit8).astype(np.int64)
    return labels


def initialize_filter(size, scale = 1.0):
    stddev = scale/np.sqrt(np.prod(size))
    return np.random.normal(loc = 0, scale=stddev, size = size)


def initialize_weight(size):
    return np.random.standard_normal(size=size)*0.01


def nanagrmax(arr):
    idx = np.nanargmax(arr)
    idxs = np.unravel_index(idx, arr.shape)
    return idxs


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_s=1, pool_f=2, pool_s=2):
    conv1 = convolution(image, f1, b1, conv_s)
    conv1[conv1 <= 0] = 0

    conv2 = convolution(conv1, f2, b2, conv_s)
    conv2[conv2 <= 0] = 0

    pool_layer = max_pool(conv2, pool_f, pool_s)
    (nf2, dim2, _) = pool_layer.shape
    fc = pool_layer.reshape((nf2*dim2*dim2,1))

    z = w3.dot(fc) + b3
    z[z <= 0] = 0

    out = w4.dot(z) + b4
    probs = soft_max(out)

    return np.argmax(probs), np.max(probs)