# Provide baseline (hmmlearn) results for different data sizes
# Look at:
#  co-occurrence loss
#  loglikelihood
#  time

import numpy as np
import argparse
import time
import pickle
import itertools
from hmmlearn import hmm
from pathlib import Path
from scipy.special import erf
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from ssm.util import find_permutation

np.random.seed(42)

# Parse arguments:
#   - T  ?
#   - n
#   - dim ?


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=3,
                        help="number of hidden states")
    parser.add_argument("-d", type=int, default=1,
                        help="dimensionality of observed values")
    parser.add_argument("-T", type=int, default=1000,
                        help="length of observations")
    args = parser.parse_args()
    return args.n, args.d, args.T


def get_params(d_, n_):
    transmat_ = np.random.uniform(0, 1, (n_, n_))
    transmat_ /= transmat_.sum(axis=1)[:, np.newaxis]
    startprob_ = compute_stationary(transmat_)
    means_ = np.random.uniform(-100, 100, (n_, d_))
    covars_ = np.random.uniform(0.5, 4, (n_, d_))
    return startprob_, transmat_, means_, covars_


# Sample from model
def sample(n_, T_, startprob_, transmat_, means_, covars_):
    hmm_gt = hmm.GaussianHMM(n_components=n_, covariance_type="diag")
    hmm_gt.startprob_ = startprob_
    hmm_gt.transmat_ = transmat_
    hmm_gt.means_ = means_
    hmm_gt.covars_ = covars_
    results = [hmm_gt.sample(T_ // 10) for _ in range(10)]
    return np.concatenate([r[0] for r in results]), \
           np.concatenate([r[1] for r in results]), \
           np.array([T_ // 10 for _ in range(10)])


def compute_stationary(M, verbose=True):
    eigval, eigvec = np.linalg.eig(M.T)
    idx = np.asarray(np.isclose(eigval, [1.])).nonzero()[0]
    if idx.size < 1:
        raise Exception("No Eigenvalue 1")
    elif idx.size > 1 and verbose:
        print("Warning: Multiple vectors corresponding to eigenvalue 1.: %s" % str(idx))
    M_stationary = eigvec[:, idx[0]].real
    M_stationary = M_stationary / np.sum(M_stationary)
    return M_stationary


def relu(x):
    return x * (x > 0)


def compute_loss(nodes, splits, n_, omega_gt, means_, covars_, A_):  # TODO rename arguments
    if d == 1:  # popraw wizualnie
        B_scalars_tmp = .5 * (
                1 + erf((nodes - np.transpose(a=means_)) / (
                relu(np.transpose(a=covars_[:, :, 0])) + 1e-10) / np.sqrt(2)))

        B_scalars = np.transpose(a=B_scalars_tmp[1:, :] - B_scalars_tmp[:-1, :])
    if d == 2:
        B_scalars_tmp = np.prod(.5 * (
                1 + erf((np.expand_dims(nodes, axis=-1) - np.expand_dims(
            np.transpose(a=means_), axis=0)) / (
                                relu(covars_) + 1e-10) / np.sqrt(2))), axis=1)
        B_scalars_tmp_wide = np.reshape(B_scalars_tmp, (*[n.shape[0] for n in splits], n_))
        B_scalars = np.transpose(a=np.reshape(
            B_scalars_tmp_wide[:-1, 1:, :] - B_scalars_tmp_wide[:-1, :-1, :] - B_scalars_tmp_wide[1:, 1:,
                                                                               :] + B_scalars_tmp_wide[1:, :-1, :],
            (-1, n_)))

    A_stationary = compute_stationary(A_, verbose=False)

    theta = A_ * A_stationary[:, None]
    omega = np.matmul(np.transpose(a=B_scalars), np.matmul(theta, B_scalars))
    loss_cooc = np.sum(np.abs(omega_gt - omega)) / 2
    return loss_cooc


def provide_nodes(n_, Y_train):
    kmeans = KMeans(n_).fit(Y_train)
    dtree = DecisionTreeClassifier().fit(Y_train, kmeans.labels_)

    splits = np.concatenate([dtree.tree_.feature.reshape(-1, 1), dtree.tree_.threshold.reshape(-1, 1)], axis=1)
    splits = np.concatenate([splits, np.array([[i, fun(Y_train[:, i])] for i, fun in itertools.product(range(Y_train.shape[1]), [
        lambda x: np.min(x) - 1e-3, lambda x: np.max(x) + 1e-3])])])
    splits = splits[splits[:, 0] >= 0]

    nodes_x = [np.sort(splits[splits[:, 0] == float(i), 1]) for i in np.unique(splits[:, 0])]
    nodes = np.array([t for t in itertools.product(*nodes_x)])

    splits = nodes_x  # number  of splits  on each axis
    discrete_nodes = nodes.astype('float32')
    discrete_observables = [n.shape[0] - 1 for n in nodes_x]


    indexes = np.arange(np.prod(discrete_observables)).reshape(discrete_observables)  # .transpose()
    Y_train_disc = np.array([indexes[i] for i in
                       zip(*[(Y_train[:, j].reshape(-1, 1) > splits[j].reshape(1, -1)).sum(axis=1) - 1 for j in
                             range(len(splits))])])

    return nodes, nodes_x, Y_train_disc


def _lengths_iterator(seqs, lengths):
    n_seqs = len(lengths)
    left, right = 0, 0

    for i in range(len(lengths)):
        right += lengths[i]
        yield seqs[left:right]
        left += lengths[i]


def empirical_coocs(seqs, m, lengths=None):
    freqs = np.zeros((m, m))
    seq_iterator = seqs
    if lengths is not None:
        seq_iterator = _lengths_iterator(seqs, lengths)

    for seq in seq_iterator:

        if seq.shape[0] <= 1:  # no transitions
            continue

        seq = seq.reshape(-1)

        seq_pairs = np.dstack((seq[:-1], seq[1:]))
        seq_pairs, counts = np.unique(seq_pairs, return_counts=True, axis=1)
        seq_pre, seq_suc = [arr.flatten() for arr in np.dsplit(seq_pairs, 2)]
        freqs[seq_pre, seq_suc] += counts

    return freqs, freqs / np.sum(freqs)


def eval_model(n_, Y_train, X_train, lengths_):
    nodes, splits, Y_disc = provide_nodes(n_, Y_train)
    _, omega_gt = empirical_coocs(Y_disc.reshape(-1, 1), np.max(Y_disc) + 1, lengths=lengths_)

    model = hmm.GaussianHMM(n_components=n_, covariance_type='full')
    start = time.time()
    model.fit(Y_train, lengths_)
    end = time.time()
    ll = model.score(Y_train, lengths_)
    states = model.predict(Y_train, lengths_)
    acc = (find_permutation(states, X_train)[states] == X_train).mean()
    loss = compute_loss(nodes, splits, n_, omega_gt, model.means_, model.covars_, model.transmat_)
    return ll, loss, acc, end - start

if __name__ == "__main__":
    n, d, T = parse_args()
    t = time.localtime()
    result_dir = f"../../data/benchramk_artificial/n_{n}_T{T}_d{d}_{t.tm_year}-{t.tm_mon}-{t.tm_mday}"
    Path(result_dir).mkdir(exist_ok=True, parents=True)
    startprob, transmat, means, covars = get_params(d, n)
    Y, X, lengths = sample(n, T, startprob, transmat, means, covars)
    ll, loss, acc, dur = eval_model(n, Y, X, lengths)

    experiment = dict(
        startprob=startprob, transmat=transmat, means=means, covars=covars,
        n=n, d=d, T=T, Y=Y, X=X,
        loglikelihood=ll, omega_loss=loss, accuracy=acc, time=dur
    )
    print(experiment)
    with open(f"{result_dir}/result.pkl", 'wb') as f:
        pickle.dump(experiment, f)
