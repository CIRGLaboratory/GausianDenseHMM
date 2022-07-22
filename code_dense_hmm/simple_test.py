import itertools
import json
import numpy as np
from hmmlearn import hmm
from models import StandardHMM, DenseHMM
import time
from tqdm import tqdm
from ssm.util import find_permutation

np.random.seed(2022)

def prepare_params(n, v, A_stat=False):
    pi = np.random.uniform(size=n)
    pi /= pi.sum()
    if A_stat:
        A = np.repeat(pi[np.newaxis,:], n, axis=0)
    else:
        A = np.exp(np.random.uniform(0, 5, size=(n, n)))
        A /= A.sum(axis=1)[:, np.newaxis]

    B = np.exp(np.random.uniform(0, 5, size=(n,  v)))
    B /= B.sum(axis=1)[:, np.newaxis]
    return pi, A, B


def my_hmm_sampler(pi, A, B, T):
    n = pi.shape[0]
    v = B.shape[1]
    X = [np.random.choice(np.arange(n), 1, replace=True, p=pi)]
    for t in range(T - 1):
        X.append(np.random.choice(np.arange(n), 1, replace=True, p=A[X[t][0], :]))
    Y = np.concatenate([np.random.choice(np.arange(v), 1, replace=True, p=B[s[0], :]) for s in X]).reshape(-1, 1)
    return X, Y


def experiment(n, m, T, s,  l, A_stat=False):
    pi, A, B = prepare_params(n, m, A_stat)
    data = [my_hmm_sampler(pi, A, B, T) for _ in range(s)]
    X_true = np.concatenate([x[1] for x in data])
    lenghts = [len(x[1]) for x in data]
    Y_true = np.concatenate([np.concatenate(y[0]) for y in data])
    assert(np.unique(X_true).shape[0] == m)

    standard_acc = []
    standard_timer = []
    dense_acc = []
    dense_timer = []
    hmml_acc = []
    hmml_timer = []
    for _ in tqdm(range(10), desc="HMM"):
        A_init = np.exp(np.random.uniform(0, 5, size=(n, n)))
        A_init /= A_init.sum(axis=1)[:, np.newaxis]


        start = time.perf_counter()
        standardhmm = StandardHMM(n, em_iter=1000, init_params="se")
        standardhmm.transmat_ = A_init
        standardhmm.fit(X_true, lenghts)
        preds = np.concatenate([standardhmm.predict(x[1].transpose()) for x in data])
        standard_timer.append(time.perf_counter() - start)
        perm = find_permutation(preds, Y_true)
        standard_acc.append((Y_true == np.array([perm[i] for i in preds])).mean())

        start = time.perf_counter()
        densehmm = DenseHMM(n, init_params="se",  mstep_config={"l_uz":  l, "l_vw": l})
        densehmm.transmat_ = A_init
        densehmm.fit_coocs(X_true, lenghts)
        preds = np.concatenate([densehmm.predict(x[1].transpose()) for x in data])
        perm = find_permutation(preds, Y_true)
        dense_timer.append(time.perf_counter() - start)
        dense_acc.append((Y_true == np.array([perm[i] for i in preds])).mean())

        start = time.perf_counter()
        hmml = hmm.MultinomialHMM(n, n_iter=1000, init_params="se")
        hmml.transmat_ = A_init
        hmml.fit(X_true, lenghts)
        preds = np.concatenate([hmml.predict(x[1].transpose()) for x in data])
        hmml_timer.append(time.perf_counter() - start)
        perm = find_permutation(preds, Y_true)
        hmml_acc.append((Y_true == np.array([perm[i] for i in preds])).mean())

    return {"standard_acc": standard_acc,
            "standard_time":  standard_timer,
            # "ssm_acc": ssm_acc,
            # "ssm_time":  ssm_time,
            "dense_acc": dense_acc,
            "dense_time": dense_timer,
            "hmml_acc": hmml_acc,
            "hmml_time": hmml_timer,
            "pi": pi,
            "A": A,
            "B": B}


def run_experiments():
    results = []
    for n, v, T, A_stat, l in tqdm(itertools.product([2, 3, 4, 8],  [5, 10, 20], [10, 100, 1000],  [True, False],  [3,  4,  5]),  desc="EXPERIMENT"):
        tmp = experiment(n, v, T, 100, l, A_stat)
        results.append({**tmp, "n": n,  "v":  v, "T": T,  "A_stat": A_stat})
    return results

if __name__ == "__main__":
    result = run_experiments()
    with open("experiment_result_22-07-22.json",  "w") as f:
        json.dump(result, f)