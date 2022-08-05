import numpy as np
import itertools
from hmmlearn import hmm
from models_gaussian import StandardGaussianHMM, GaussianDenseHMM, HMMLoggingMonitor
import time
from tqdm import tqdm
from ssm.util import find_permutation
import pickle
import json
from pathlib import Path
import wandb
from utils import dtv, permute_embeddings, compute_stationary
import scipy.stats as stats
import matplotlib.pyplot as plt

np.random.seed(2022)

simple_model = {"mu": 10,
                "sigma": 1}

complicated_model = {"mu": 5,
                     "sigma": 2}

data_sizes = [(100, 400, 4)]  # (s, T, n)

ls = (2, 3, 4,  5)
ms = (4, 8, 12, 64, 128)

mstep_cofigs_em = [{"em_lr": 0.0001, "em_epochs": 10},
                   {"em_lr": 0.0001, "em_epochs": 25},
                   {"em_lr": 0.0001, "em_epochs": 50},
                   {"em_lr": 0.001, "em_epochs": 10},
                   {"em_lr": 0.001, "em_epochs": 25},
                   {"em_lr": 0.01, "em_epochs": 5},
                   {"em_lr": 0.01, "em_epochs": 10},
                   {"em_lr": 0.05, "em_epochs": 10},
                   {"em_lr": 0.1, "em_epochs": 5}]

mstep_cofigs_cooc = [{"cooc_lr": 0.001, "cooc_epochs": 100000},
                     {"cooc_lr": 0.01, "cooc_epochs": 100000},
                     {"cooc_lr": 0.1, "cooc_epochs": 100000},
                     {"cooc_lr": 0.25, "cooc_epochs": 100000}]

EM_ITER = 100
TOLERANCE = 1e-4


def prepare_params(n, simple_model=True):
    A = np.exp(np.random.uniform(0, 5, size=(n, n)))
    A /= A.sum(axis=1)[:, np.newaxis]

    pi = compute_stationary(A)

    if simple_model:
        mu = np.arange(n) * 10
        sigma = np.ones(shape=n)
    else:
        mu = np.random.uniform(0, n * 3, size=n)
        sigma = np.random.uniform(.5, 1.75, size=n)
    return pi, A, mu, sigma


def my_hmm_sampler(pi, A, mu, sigma, T):
    n = pi.shape[0]
    X = [np.random.choice(np.arange(n), 1, replace=True, p=pi)]
    for t in range(T - 1):
        X.append(np.random.choice(np.arange(n), 1, replace=True, p=A[X[t][0], :]))
    Y = np.concatenate([np.random.normal(mu[s[0]], sigma[s[0]], 1) for s in X]).reshape(-1, 1)
    return X, Y


def provide_log_info(pi, A, mu, sigma, X_true, model, time_tmp, perm, preds_perm, mstep_cofig=None, embeddings=None):
    return [dict(time=time_tmp,
                 acc=(X_true == preds_perm).mean(),
                 d_tv_A=dtv(model.transmat_, A[perm, :][:, perm]),
                 d_tv_pi=dtv(model.startprob_.reshape(1, -1), pi[perm].reshape(1, -1)),
                 MAE_mu=abs(mu[perm] - model.means_[:, 0]).mean(),
                 MAE_sigma=abs(sigma[perm] - model.covars_[:, 0, 0]).mean(),
                 mu_est=model.means_.tolist(),
                 sigma_est=model.covars_.tolist(),
                 A_est=model.transmat_.tolist(),
                 pi_est=model.startprob_.tolist(),
                 preds=preds_perm.tolist(),
                 mstep_cofig=mstep_cofig,
                 embeddings=permute_embeddings(embeddings, perm))]


def init_model(model, A_init, pi_init, m_init, c_init):
    model.transmat_ = A_init
    model.startprob_ = pi_init
    model.means_ = m_init
    model._covars_ = c_init


def predict_permute(model, data, X_true):
    preds = np.concatenate([model.predict(x[1]) for x in data])
    perm = find_permutation(preds, X_true)
    return np.array([perm[i] for i in preds]), perm


def em_scheduler(max_lr, it):
    if it <= np.ceil(EM_ITER / 3):
        return max_lr * np.cos(3 * (np.ceil(EM_ITER / 3) - it) * np.pi * .33 / EM_ITER)
    else:
        return max_lr * np.cos((it - np.ceil(EM_ITER / 3)) * np.pi * .66 / EM_ITER) ** 3


def init_experiment(dsize, simple_model):
    s = dsize[0]
    T = dsize[1]
    n = dsize[2]
    pi, A, mu, sigma = prepare_params(n, simple_model)

    data = [my_hmm_sampler(pi, A, mu, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    lengths = [len(x[1]) for x in data]

    result = {
        "number_of_sequences": s,
        "sequence_length": T,
        "number_of_hidden_states": n,
        "pi": pi.tolist(),
        "A": A.tolist(),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "simple_model": simple_model,
        "data": [X_true.tolist(), Y_true.tolist(), lengths],
        "hmmlearn_runs": [],
        "standard_gaussian_runs": [],
        "dense_em_runs": [],
        "dense_cooc_runs": []
    }

    true_values = {
        "states": X_true,
        "transmat": A,
        "startprob": pi,
        "means": mu,
        "covars": sigma
    }

    wandb_params = {
        "init": {
            "project": "gaussian-dense-hmm",
            "entity": "cirglaboratory",
            "save_code": True,
            "group": f"benchmark-{t.tm_year}-{t.tm_mon}-{t.tm_mday}",
            "job_type": f"n={n}-s={s}-T={T}-simple={simple_model}",
            "name": f"PDFs",
            "reinit": True
        },
        "config": {
            "n": n,
            "s": s,
            "T": T,
            "model": None,
            "m": None,
            "l": None,
            "lr":  0,
            "em_epochs":  0,
            "em_iter": EM_ITER,
            "cooc_epochs": 0,
            "epochs": 0,
            "simple_model": simple_model
        }
    }

    wandb.init(**wandb_params["init"], config=wandb_params["config"])

    x = np.linspace(min(mu) - 3 * max(sigma), max(mu) + 3 * max(sigma), 10000)
    for i in range(n):
        plt.plot(x, stats.norm.pdf(x, mu[i], sigma[i]), label=str(i))
    plt.title(f"Normal PDFs n={n}-s={s}-T={T}-simple={simple_model}")
    wandb.log({"Normal densities": wandb.Image(plt)})
    plt.close()

    plt.plot([em_scheduler(1, it) for it in range(EM_ITER)])
    plt.title("Learning rate schedule")
    wandb.log({"LR schedule": wandb.Image(plt)})
    plt.close()

    return s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, data


def run_experiment(results_dir, simple_model=True):
    dsize = data_sizes[0]
    s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, data = init_experiment(
        dsize, simple_model)

    # GaussianHMM - custom implementation
    # wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}", "name": "standard"})
    # wandb_params["config"].update(
    #     dict(model="standard", m=0, l=0, lr=0, em_epochs=0, em_iter=EM_ITER, cooc_epochs=0, epochs=0))
    # wandb.init(**wandb_params["init"], config=wandb_params["config"])
    #
    # hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
    #                                 wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
    #                                 log_config={'metrics_after_convergence': True})
    #
    # standardhmm = StandardGaussianHMM(n, em_iter=EM_ITER, covariance_type='diag', init_params="stmc", params="stmc",
    #                                   early_stopping=True, logging_monitor=hmm_monitor)
    #
    # start = time.perf_counter()
    # standardhmm.fit(Y_true, lengths)
    # time_tmp = time.perf_counter() - start
    #
    # preds_perm, perm = predict_permute(standardhmm, data, X_true)
    # result['standard_gaussian_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
    #                                             standardhmm, time_tmp, perm, preds_perm)

    # GaussianDenseHMM - custom implementation with coocurrences-based fit
    for mstep_cofig, l, m in itertools.product(mstep_cofigs_cooc, ls, ms):

        wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                     "name": f"dense--l={l}-lr={mstep_cofig['cooc_lr']}-epochs={mstep_cofig['cooc_epochs']}"})
        wandb_params["config"].update(dict(model="dense_cooc_log_abs", m=m, l=l, lr=mstep_cofig['cooc_lr'], em_epochs=0,
                                           em_iter=0, cooc_epochs=mstep_cofig['cooc_epochs'], epochs=mstep_cofig['cooc_epochs'],
                                           loss_type='abs_log', "scheduler"=True))

        wandb.init(**wandb_params["init"], config=wandb_params["config"])
        hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                        wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                        log_config={'metrics_after_convergence': True})
        for _ in range(3):
            densehmm = GaussianDenseHMM(n, mstep_config={**mstep_cofig, "l_uz": l, 'loss_type': 'abs_log'},
                                        covariance_type='diag', em_iter=EM_ITER, logging_monitor=hmm_monitor,
                                        init_params="stmc", params="stmc", early_stopping=False, opt_schemes={"cooc"},
                                        discrete_observables=m)

            start = time.perf_counter()
            try:
                densehmm.fit_coocs(Y_true, lengths)
            except:
                0
            time_tmp = time.perf_counter() - start

            preds_perm, perm = predict_permute(densehmm, data, X_true)

            result['dense_cooc_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
                                                          densehmm, time_tmp, perm, preds_perm,
                                                          {**mstep_cofig, "l_uz": l},
                                                          embeddings=densehmm.get_representations())


    # # GaussianDenseHMM - custom implementation
    # for mstep_cofig, l in itertools.product(mstep_cofigs_em, ls):
    #
    #     wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
    #                                  "name": f"dense--l={l}-lr={mstep_cofig['em_lr']}-epochs={mstep_cofig['em_epochs']}"})
    #     wandb_params["config"].update(dict(model="dense_em", m=0, l=l, lr=mstep_cofig['em_lr'],
    #                                        em_epochs=mstep_cofig['em_epochs'], em_iter=EM_ITER,
    #                                        cooc_epochs=0, epochs=EM_ITER * mstep_cofig['em_epochs']))
    #
    #     wandb.init(**wandb_params["init"], config=wandb_params["config"])
    #     hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
    #                                     wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
    #                                     log_config={'metrics_after_convergence': True})
    #     for _ in range(3):
    #         densehmm = GaussianDenseHMM(n, mstep_config={**mstep_cofig, "l_uz": l, "scheduler": em_scheduler},
    #                                     covariance_type='diag', em_iter=EM_ITER, logging_monitor=hmm_monitor,
    #                                     init_params="stmc", params="stmc", early_stopping=False)
    #
    #         start = time.perf_counter()
    #         densehmm.fit(Y_true, lengths)
    #         time_tmp = time.perf_counter() - start
    #
    #         preds_perm, perm = predict_permute(densehmm, data, X_true)
    #
    #         result['dense_em_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
    #                                                     densehmm, time_tmp, perm, preds_perm,
    #                                                     {**mstep_cofig, "l_uz": l},
    #                                                     embeddings=densehmm.get_representations())

    with open(f"{results_dir}/s={s}_T={T}_n={n}_simple={simple_model}.json", "w") as f:
        json.dump(result, f, indent=4)

    return None


if __name__ == "__main__":
    start = time.perf_counter()
    t = time.localtime()
    results_dir = f'gaussian_dense_hmm_benchmark/basic-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'
    Path(results_dir).mkdir(exist_ok=True, parents=True)

    run_experiment(results_dir, simple_model=True)
    run_experiment(results_dir, simple_model=False)
    print("DONE. All computations took:", time.perf_counter() - start, "seconds.")
