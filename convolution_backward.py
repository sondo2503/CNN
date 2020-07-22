import numpy as np


def convolution_backward(dconv_prev, conv_in, filters, s):
    (n_f, n_c, f, _) = filters.shape
    (_, orig_dim, _) = conv_in.shape
    d_out = np.zeros(conv_in.shape)
    d_filter = np.zeros(filters.shape)
    d_bias = np.zeros((n_f,1))
    for curr_f in range(n_f):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                d_filter += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                d_out[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filters[curr_f]
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        d_bias[curr_f] = np.sum(dconv_prev[curr_f])

    return d_out, d_filter, d_bias

