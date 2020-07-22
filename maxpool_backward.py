import numpy as np
from utilitys import *


def maxpool_backward(d_pool, orig, f, s):
    (n_c, orig_dim, _) = orig.shape
    d_out = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                (a, b) = nanagrmax(orig[curr_c, curr_y:curr_y+f, curr_x:curr_x+f])
                d_out[curr_c, curr_y+a, curr_x+b] = d_pool[curr_c, out_x, out_y]

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return d_out

