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

data_sizes = [(100, 1000, 15)]  # (s, T, n)

ls = (2, 3, 4,  5,  8,  10)
ms = (8, 12, 20, 50,  75, 100,  200)

# mstep_cofigs = [{"em_lr": 0.0001, "em_epochs": 10},
#                 {"em_lr": 0.0001, "em_epochs": 25},
#                 {"em_lr": 0.0001, "em_epochs": 50},
#                 {"em_lr": 0.001, "em_epochs": 10},
#                 {"em_lr": 0.001, "em_epochs": 25},
#                 {"em_lr": 0.01, "em_epochs": 5},
#                 {"em_lr": 0.01, "em_epochs": 10}]

mstep_cofigs = [#{"cooc_lr": 0.00001, "cooc_epochs": 10000},  # Za mały LR
                # {"cooc_lr": 0.0001, "cooc_epochs": 10000},
                # {"cooc_lr": 0.001, "cooc_epochs": 10000},
                {"cooc_lr": 0.01, "cooc_epochs": 100000},
                {"cooc_lr": 0.1, "cooc_epochs": 100000}]

EM_ITER = 50
TOLERANCE = 1e-4


def prepare_params(n, simple_model=True):
    A = np.exp(np.random.uniform(0, 5, size=(n, n)))
    A /= A.sum(axis=1)[:, np.newaxis]

    pi = compute_stationary(A)

    if simple_model:
        mu = np.arange(n) * 10
        sigma = np.ones(shape=n)
    else:
        mu = np.random.uniform(0, n*3, size=n)
        sigma = np.random.uniform(.5, 1.75, size=n)
    return pi, A, mu, sigma


def my_hmm_sampler(pi, A,  mu, sigma, T):
    n = pi.shape[0]
    X = [np.random.choice(np.arange(n), 1, replace=True, p=pi)]
    for t in range(T - 1):
        X.append(np.random.choice(np.arange(n), 1, replace=True, p=A[X[t][0], :]))
    Y = np.concatenate([np.random.normal(mu[s[0]], sigma[s[0]], 1) for s in X]).reshape(-1, 1)
    return X, Y


def provide_log_info(pi, A, mu, sigma, X_true, model,  time_tmp, perm, preds_perm, mstep_cofig=None, embeddings=None):
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
        "gaussian_dense_runs": []
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
            "group": f"benchmark-{t.tm_year}-{t.tm_mon}-{t.tm_mday}-test",  # v2
            "job_type": f"n={n}-s={s}-T={T}-simple={simple_model}",
            "name": f"PDFs",
            "reinit": True
        },
        "config": {
            "n": n,
            "s": s,
            "T": T,
            "model": None
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
    for rep in tqdm(range(1), desc=f"Experiment"):
        dsize = data_sizes[0]
        s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, data = init_experiment(dsize, simple_model)

        # GaussianHMM - hmmlearn implementation
        hmml = hmm.GaussianHMM(n, n_iter=EM_ITER, covariance_type='diag', init_params="", params="stmc")
        hmml._init(Y_true)
        # Randomly chosen initial parameters
        A_init = hmml.transmat_
        pi_init = hmml.startprob_
        m_init = hmml.means_
        c_init = hmml._covars_

        # start = time.perf_counter()
        # hmml.fit(Y_true, lengths)
        # time_tmp = time.perf_counter() - start
        #
        # preds_perm, perm = predict_permute(hmml, data, X_true)
        # result['hmmlearn_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
        #                                             hmml,  time_tmp, perm, preds_perm)

        # GaussianHMM - custom implementation
        # wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}", "name": "standard"})
        # wandb_params["config"].update({"model": "standard"})
        # wandb.init(**wandb_params["init"], config=wandb_params["config"])
        #
        # hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
        #                                 wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
        #                                 log_config={'metrics_after_convergence': True})
        #
        # standardhmm = StandardGaussianHMM(n, em_iter=EM_ITER, covariance_type='diag', init_params="", params="stmc",
        #                                   early_stopping=False, logging_monitor=hmm_monitor)
        # init_model(standardhmm, A_init, pi_init, m_init, c_init)
        #
        # start = time.perf_counter()
        # standardhmm.fit(Y_true, lengths)
        # time_tmp = time.perf_counter() - start
        #
        # preds_perm, perm = predict_permute(standardhmm, data, X_true)
        # result['standard_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
        #                                             standardhmm, time_tmp, perm, preds_perm)

        # # GaussianDenseHMM - custom implementation
        for mstep_cofig, l,  m in itertools.product(mstep_cofigs, ls, ms):
            # # # allow only embeddings of size smaller or equal the number of hidden states
            # # if l > n:
            # #     continue
            #
            # wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
            #                              "name": f"dense--l={l}-lr={mstep_cofig['em_lr']}-epochs={mstep_cofig['em_epochs']}"})
            # wandb_params["config"].update({**mstep_cofig, "l": l, "model": "dense"})
            # wandb.init(**wandb_params["init"], config=wandb_params["config"])
            # hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
            #                                 wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
            #                                 log_config={'metrics_after_convergence': True})
            #
            # densehmm = GaussianDenseHMM(n, mstep_config={**mstep_cofig, "l_uz": l, "em_scheduler": em_scheduler},
            #                             covariance_type='diag', em_iter=EM_ITER, logging_monitor=hmm_monitor,
            #                             init_params="", params="stmc", early_stopping=False)
            # init_model(densehmm, A_init, pi_init, m_init, c_init)
            #
            # start = time.perf_counter()
            # densehmm.fit(Y_true, lengths)
            # time_tmp = time.perf_counter() - start
            #
            # preds_perm, perm = predict_permute(densehmm, data, X_true)
            #
            # result['dense_em_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
            #                                             densehmm, time_tmp, perm, preds_perm,
            #                                             {**mstep_cofig, "l_uz": l},
            #                                             embeddings=densehmm.get_representations())
            wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={T}-simple={simple_model}",
                                         "name": f"dense--l={l}-lr={mstep_cofig['cooc_lr']}-epochs={mstep_cofig['cooc_epochs']}"})
            wandb_params["config"].update({**mstep_cofig, "l": l, "model": "dense"})
            wandb.init(**wandb_params["init"], config=wandb_params["config"])
            hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                            wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                            log_config={'metrics_after_convergence': True})

            densehmm = GaussianDenseHMM(n, mstep_config={**mstep_cofig, "l_uz": l, "em_scheduler": em_scheduler},
                                        covariance_type='diag', em_iter=EM_ITER, logging_monitor=hmm_monitor,
                                        init_params="", params="stmc", early_stopping=False, opt_schemes={"cooc"},
                                        discrete_observables=m)
            init_model(densehmm, A_init, pi_init, m_init, c_init)

            start = time.perf_counter()
            densehmm.fit_coocs(Y_true, lengths)
            time_tmp = time.perf_counter() - start

            preds_perm, perm = predict_permute(densehmm, data, X_true)

            result['gaussian_dense_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
                                                        densehmm, time_tmp, perm, preds_perm,
                                                        {**mstep_cofig, "l_uz": l},
                                                        embeddings=densehmm.get_representations())

        with open(f"{results_dir}/s={s}_T={T}_n={n}_simple={simple_model}_{rep}.json", "w") as f:
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