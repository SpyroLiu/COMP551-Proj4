"""
"""

import numpy as np
import scipy.linalg as linalg


def caratheodory(P, u=None, tol=1e-8, dtype=np.float32):
    (n, d) = P.shape

    if u is None:
        u = np.ones(n, dtype)
    else:
        u = u.astype(dtype)

    u.reshape(-1,1)

    mask = u != 0
    n = np.sum(mask)

    while n > d+1:
        v = linalg.svd((P[mask][:-1] - P[mask][-1]).T, full_matrices=True)[2][-1]
        v = np.append(v, -np.sum(v))

        positive = v > 0
        u[mask] = u[mask] - np.min(u[mask][positive]/v[positive]) * v
        u[np.isclose(u, 0, atol=tol)] = 0
        # unrolled recursive step
        mask = u != 0
        n = np.sum(mask)

    return u


def fast_caratheodory(P, u=None, k = None, size=None, tol=1e-8, dtype=np.float32):
    n, d = P.shape
    n_ = n # save original size

    if u is None:
        u = np.ones(n, dtype)
    else:
        u = u.astype(dtype)

    mask = u != 0
    P = P[mask]
    u = u[mask]
    idx = np.arange(n_, dtype=np.int)[mask] # idx of the computed weights

    n = np.sum(mask)

    # default fastest value for k
    if not k:
        k = 2 *d + 2
    k_ = k

    # scaling, see original implementation
    u_sum = np.sum(u)
    u /= u_sum

    if not size:
        size = d + 1
    while n > size:
        # discretize cluster count and size
        cluster_size = int(np.ceil(n/k_))
        k = int(np.ceil(n/cluster_size))
        # fill data to match cluster size
        fill = cluster_size - n % cluster_size
        if fill != cluster_size:
            P = np.append(P, np.zeros((fill, d), dtype))
            u = np.append(u, np.zeros(fill, dtype))
            idx = np.append(idx, np.zeros(fill, dtype=np.int32))
        # partition into clusters
        clusters = P.reshape(k, cluster_size, d)
        cluster_weights = u.reshape(k, cluster_size)
        cluster_idx = idx.reshape(k, cluster_size)

        # weighted means of the clusters
        means = np.einsum('ijk,ij->ik', clusters, cluster_weights)
        # call to caratheodory using weighted cluster means
        w = caratheodory(means, np.ones(k, dtype), tol, dtype)

        cluster_mask = w != 0
        P = clusters[cluster_mask].reshape(-1,d)
        u = (cluster_weights[cluster_mask] * w[cluster_mask][:, np.newaxis]).reshape(-1)
        idx = cluster_idx[cluster_mask].reshape(-1)

        n = u.shape[0]

    u_new = np.zeros(n_, dtype)
    u_new[idx] = u
    u_new *= u_sum
    return u_new


def coreset(X, y=None, weights=None, k=None, size=None, tol=1e-8, dtype=np.float32):
    n, d = X.shape
    P = X[:,:,np.newaxis]
    P = np.einsum('ikj,ijk->ijk', P, P)
    P = P.reshape(n,d**2)

    w = np.sqrt(fast_caratheodory(P, weights, k, size, tol, dtype))
    idx_mask = w != 0

    if y is not None:
        return X[idx_mask], y[idx_mask], w[idx_mask][:,np.newaxis]
    else:
        return X[idx_mask], w[idx_mask][:,np.newaxis]



if __name__ == '__main__':
    from ablation import load_datasets
    from Booster import Caratheodory, Fast_Caratheodory
    d = load_datasets()[0][:100]
    u = np.arange(100)

    # test caratheodory
    c = caratheodory(d, u)
    assert(np.all(u == np.arange(100)))
    assert(np.allclose(d.T @ c, d.T @ u))

    # test fast_caratheodory
    c = fast_caratheodory(d,u)
    assert(np.all(u == np.arange(100)))
    assert(np.allclose(d.T @ c, d.T @ u))

    A = d[:,:-1]
    y = d[:,-1]
    C, y, w = coreset(A, y)
    S = w * C
    print((A.T @ A, S.T @ S))
    assert(np.allclose(A.T @ A, S.T @ S))
