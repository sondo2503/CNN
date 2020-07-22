import numpy as np


def convolution(img, filters, bias, s=1):
    (n_f, n_c_f, f, _) = filters.shape
    n_c, in_dim, _ = img.shape
    out_dim = int((in_dim - f) / s + 1)
    assert n_c == n_c_f
    out = np.zeros((n_f, out_dim, out_dim))
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= in_dim:
            curr_x = out_x = 0
            while curr_x + f <= in_dim:
                out[curr_f, out_y, out_x] = np.sum(filters[curr_f] * img[:, curr_y:curr_y + f, curr_x:curr_x + f]) + bias[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return out


ex_img = np.random.rand(3, 32, 32)
ex_bias = np.zeros(2, )
ex_filters = np.zeros((2, 3, 3, 3))
ex_filters[0, :, :] = np.array([[[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]]])
ex_filters[1, :, :] = np.array([[[1, 1, 1],
                                 [0, 0, 0],
                                 [1, 1, 1]]])
print(ex_img, sep="    ")
print(ex_img.shape)
cnn = convolution(ex_img, ex_filters, bias=ex_bias, s=1)
print(cnn.shape)


