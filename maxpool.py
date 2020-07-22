import numpy as np


def max_pool(img, f=2, s=2):
    n_c, h_prev, w_prev = img.shape
    h = int((h_prev-f)/s)+1
    w = int((w_prev-f)/s)+1
    down_sampled = np.zeros((n_c, h, w))
    for i in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= h_prev:
            curr_x = out_x = 0
            while curr_x + f <= w_prev:
                down_sampled[i, out_y, out_x] = np.max(img[i, curr_y:curr_y+f, curr_x:curr_x+f])
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
    return down_sampled


ex_img = np.random.rand(3, 32, 32)
pooled = max_pool(ex_img, f=2, s=2)
print(pooled.shape)

