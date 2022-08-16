import numpy as np
import itertools
from hmmlearn import hmm
from models_gaussian import StandardGaussianHMM, GaussianDenseHMM, HMMLoggingMonitor, DenseHMMLoggingMonitor
from models_gaussian_A import GaussianDenseHMM as CoocHMM
import time
from tqdm import tqdm
from ssm.util import find_permutation
import pickle
import joblib
import json
from pathlib import Path
import wandb
from utils import dtv, permute_embeddings, compute_stationary
import scipy.stats as stats
import matplotlib.pyplot as plt
import optuna
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import multiprocessing as mp
from celluloid import Camera
import matplotlib.cm as cm
import argparse
from collections import  defaultdict as dd
from eval_utils import *


np.random.seed(2022)

t = time.localtime()
RESULT_DIR = f'gaussian_dense_hmm_benchmark/eval-cooc-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'

data_sizes = [  # (s, T, n)
    (100, 100, 8)
]

def run_experiment(dsize, simple_model=True,  l_fixed=True):
    ## setup

    best_result = {}
    s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, _, em_scheduler = init_experiment(dsize,
                                                                                                          simple_model)
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (mu[1:] + mu[:-1]) / 2,
                            np.array([Y_true.max(), np.infty])])
    m = nodes.shape[0] - 1

    ## Tune hyper-parameters
    l = np.ceil(n / 3) if l_fixed else None
    objectives = dict(
        cooc=lambda trial: objective(trial, n, m, CoocHMM, HMMLoggingMonitor, Y_true, lengths, mu, em_scheduler, l=l),
        dense=lambda trial: objective(trial, n, m, GaussianDenseHMM, DenseHMMLoggingMonitor, Y_true, lengths, mu,
                                      em_scheduler, l=l),
        dense_em=lambda trial: objective(trial, n, m, GaussianDenseHMM, DenseHMMLoggingMonitor, Y_true, lengths, mu,
                                         em_scheduler, alg="em", l=l)
    )

    best_params = dict()
    for name in ["cooc",  "dense", "dense_em"]:
        study = optuna.create_study(directions=['maximize', 'minimize'])
        study.optimize(objectives[name], n_trials=N_TRIALS)
        best_params[name] = study.best_params
        if l_fixed:
            best_params[name]["l_param"] = l
        with open(f"{RESULT_DIR}/optuna_{name}_s{s}_T{T}_n{n}_simple_model{simple_model}_l{l_fixed}.pkl", "wb") as f:
            joblib.dump(study, f)

    ## Evaluate models

    #  prepare  new data
    data = [my_hmm_sampler(pi, A, mu, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    lengths = [len(x[1]) for x in data]

    true_values = {
        "states": X_true,
        "transmat": A,
        "startprob": pi,
        "means": mu,
        "covars": sigma
    }

    # HMMlearn
    best_result["HMMlearn"] = list()
    wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                 "name": f"HMMlearn"})
    wandb_params["config"].update(dict(model="HMMlearn", m=0, l=0, lr=0,
                                       em_iter=em_iter(n), cooc_epochs=0,
                                       epochs=0), scheduler=False, simple_model=simple_model)

    for _ in range(10):
        hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                        wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                        log_config={'metrics_after_convergence': True})
        hmm_model = hmm.GaussianHMM(n, n_iter=em_iter(n))
        hmm_model.monitor_ = hmm_monitor
        hmm_model.fit(Y_true, lengths)

        preds = hmm_model.predict(Y_true, lengths)
        perm = find_permutation(preds, X_true)

        best_result["HMMlearn"].append(
            {
            "time": time.perf_counter() - hmm_monitor._init_time,
            "logprob": hmm_model.score(Y_true, lengths),
            "acc": (X_true == np.array([perm[i] for i in preds])).mean(),
            "dtv_transmat": dtv(hmm_model.transmat_, A[perm, :][:, perm]),
            "dtv_startprob": dtv(hmm_model.startprob_, pi[perm]),
            "MAE_means": (abs(mu[perm] - hmm_model.means_[:, 0])).mean(),
            "MAE_sigma": (abs(sigma.reshape(-1)[perm] - hmm_model.covars_.reshape(-1))).mean()
            }
        )

    # Custom models
    for name in ["cooc",  "dense", "dense_em"]:
        params = best_params[name]
        best_result[name] = list()
        wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                     "name": f"name-l={params['l_param']}-lr={params['cooc_lr_param']}-epochs={params['cooc_epochs_param']}"})
        wandb_params["config"].update(dict(model="dense_cooc", m=0, l=params['l_param'], lr=params['cooc_lr_param'],
                                           em_iter=em_iter(n), cooc_epochs=params['cooc_epochs_param'],
                                           epochs=params['cooc_epochs_param']), scheduler=True, simple_model=simple_model)

        for _ in range(10):
            hmm_monitor = DenseHMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                            wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                            log_config={'metrics_after_convergence': True})
            kmeans = KMeans(n_clusters=n, random_state=0).fit(Y_true)
            nodes_tmp = np.sort(kmeans.cluster_centers_, axis=0)
            nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                                    (nodes_tmp[1:] + nodes_tmp[:-1]).reshape(-1) / 2,
                                    np.array([Y_true.max(), np.infty])])
            densehmm = GaussianDenseHMM(n, mstep_config={'cooc_epochs': params['cooc_epochs_param'],
                                                         'cooc_lr': params['cooc_lr_param'],
                                                         "l_uz": params['l_param'],
                                                         'loss_type': 'square',
                                                         'scheduler': em_scheduler},
                                        covariance_type='diag', logging_monitor=hmm_monitor, nodes=nodes,
                                        init_params="", params="stmc", early_stopping=False, opt_schemes={"cooc"},
                                        discrete_observables=m)
            densehmm.fit_coocs(Y_true, lengths)

            preds = densehmm.predict(Y_true, lengths)
            perm = find_permutation(preds, X_true)

            best_result[name].append(
                {
                "time": time.perf_counter() - hmm_monitor._init_time,
                "logprob": densehmm.score(Y_true, lengths),
                "acc": (X_true == np.array([perm[i] for i in preds])).mean(),
                "dtv_transmat": dtv(densehmm.transmat_, A[perm, :][:, perm]),
                "dtv_startprob": dtv(densehmm.startprob_, pi[perm]),
                "MAE_means": (abs(mu[perm] - densehmm.means_[:, 0])).mean(),
                "MAE_sigma": (abs(sigma.reshape(-1)[perm] - densehmm.covars_.reshape(-1))).mean()
                }
            )

    with open(f"{RESULT_DIR}/best_result_s{s}_T{T}_n{n}_simple_model{simple_model}_l{l_fixed}.json",  "w") as f:
        json.dump(best_result,  f, indent=4)
    return 0

def run_true(dsize):
    return run_experiment(dsize, simple_model=True, l_fixed=True)

def run_false(dsize):
    return run_experiment(dsize, simple_model=True, l_fixed=False)

if __name__ == "__main__":
    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)

    with mp.Pool(2) as pool:  # TODO
        pool.map(run_true, data_sizes)
        pool.map(run_false, data_sizes)
