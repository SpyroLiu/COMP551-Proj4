"""
"""

import numpy as np
import scipy.linalg as linalg


def caratheodory(P, u = None):
    if not u:
        u = np.ones(P.shape[0])
    else:
        u = u.reshape(-1)

    mask = np.nonzero(u)[0]
    (n, d) = P.shape

    while n > d+1:
        A = P[mask]
        A = (A[:-1] - A[-1]).T
        v = linalg.svd(A, full_matrices=True)[2][-1]
        v = np.append(v, -np.sum(v))

        positive = v > 0
        u[mask] = u[mask] - np.min(u[mask][positive]/v[positive]) * v
        u[np.argmin(u[mask])] = 0 # remove numerical error

        # unrolled recursive step
        mask = np.nonzero(u)[0]
        n = mask.shape[0]

    return u.reshape(-1,1)


def fast_caratheodory(P, u, k = None):
    if not u:
        u = np.ones(P.shape[0])
    else:
        u = u.reshape(-1)
    # TODO
    pass


def coreset(A, k = None):
    # TODO
    pass



if __name__ == '__main__':
    from ablation import load_datasets
    from Booster import Caratheodory
    d = load_datasets()[0][:100]
    u = np.ones((100,1))
    assert np.all(caratheodory(d,u) == Caratheodory(d,u))



