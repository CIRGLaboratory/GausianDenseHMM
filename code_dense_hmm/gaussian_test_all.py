import numpy as np
import itertools
from hmmlearn import hmm
from models_gaussian import StandardGaussianHMM, GaussianDenseHMM, HMMLoggingMonitor, DenseHMMLoggingMonitor
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

np.random.seed(2022)

simple_model = {"mu": 10,
                "sigma": 1}

complicated_model = {"mu": 5,
                     "sigma": 2}

t = time.localtime()
RESULT_DIR = f'gaussian_dense_hmm_benchmark/fit_coocs-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'

data_sizes = [  # (s, T, n)
    (100, 40, 4),
    (100, 400, 4),
    (100, 4000, 4),
    (100, 40, 8),
    (100, 400, 8),
    (100, 4000, 8),
    (100, 40, 12),
    (100, 400, 12),
    (100, 4000, 12),
    (100, 40, 20),
    (100, 400, 20),
    (100, 4000, 20),
    (100, 40, 50),
    (100, 400, 50),
    (100, 4000, 50),
    (100, 40000, 50),
    (100, 40, 100),
    (100, 400, 100),
    (100, 4000, 100),
    (100, 40000, 100)
]

EM_ITER = lambda n: 10 * n
TOLERANCE = 5e-5


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


def init_experiment(dsize, simple_model):
    s = dsize[0]
    T = dsize[1]
    n = dsize[2]
    pi, A, mu, sigma = prepare_params(n, simple_model)

    data = [my_hmm_sampler(pi, A, mu, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    lengths = [len(x[1]) for x in data]

    EM_ITER_tmp = EM_ITER(n)
    def em_scheduler(max_lr, it):
        if it <= np.ceil(EM_ITER_tmp / 3):
            return max_lr * np.cos(3 * (np.ceil(EM_ITER_tmp / 3) - it) * np.pi * .33 / EM_ITER_tmp)
        else:
            return max_lr * np.cos((it - np.ceil(EM_ITER_tmp / 3)) * np.pi * .66 / EM_ITER_tmp) ** 2

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
            "group": f"fit-coocs-benchmark-{t.tm_year}-{t.tm_mon}-{t.tm_mday}",
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
            "lr": 0,
            "em_epochs": 0,
            "em_iter": EM_ITER_tmp,
            "cooc_epochs": 0,
            "epochs": 0,
            "simple_model": simple_model,
            "scheduler": em_scheduler
        }
    }

    run = wandb.init(**wandb_params["init"], config=wandb_params["config"])

    x = np.linspace(min(mu) - 3 * max(sigma), max(mu) + 3 * max(sigma), 10000)
    for i in range(n):
        plt.plot(x, stats.norm.pdf(x, mu[i], sigma[i]), label=str(i))
    plt.title(f"Normal PDFs n={n}-s={s}-T={T}-simple={simple_model}")
    run.log({"Normal densities": wandb.Image(plt)})
    plt.close()

    plt.plot([em_scheduler(1, it) for it in range(EM_ITER_tmp)])
    plt.title("Learning rate schedule")
    run.log({"LR schedule": wandb.Image(plt)})
    plt.close()

    return s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, data, em_scheduler, run


def draw_embeddings(z, run, name="?"):
    fig = plt.figure(figsize=(5, 5))
    camera = Camera(fig)
    cmap = cm.rainbow(np.linspace(0, 1, len(z[0])))
    for z_el in z:
        if z_el.shape[1] > 1:
            plt.scatter(z_el[:, 0],  z_el[:, 1], color=cmap)
        else:
            plt.scatter(np.arange(z_el.shape[0]), z_el, color=cmap)
        camera.snap()
    plt.title(f"Embaddings trajectory:  {name}")
    animation = camera.animate()
    run.log({f"Embaddings trajectory:  {name}": wandb.Html(animation.to_html5_video())})
    plt.close()


##### OPTUNA

N_TRIALS = 64

def run_experiment(dsize, simple_model=True):
    # wandb.setup()
    # setup
    best_result = {}
    s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, _, em_scheduler, run = init_experiment(dsize,
                                                                                                          simple_model)
    # optimize hiperparams for  Dense Coocurences
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (mu[1:] + mu[:-1]) / 2,
                            np.array([Y_true.max(), np.infty])])
    m = nodes.shape[0] - 1

    def objective(trial):
        l_param = trial.suggest_int('l_param', 2, n)
        cooc_lr_param = trial.suggest_loguniform('cooc_lr_param', 1e-4, .5)
        cooc_epochs_param = trial.suggest_int('cooc_epochs_param', 10000, 100000)
        params = trial.params
        wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}-tune",
                                     "name": f"TUNE--dense--l={params['l_param']}-lr={params['cooc_lr_param']}-epochs={params['cooc_epochs_param']}"})
        wandb_params["config"].update(dict(model="dense_cooc_tune", m=0, l=params['l_param'], lr=params['cooc_lr_param'],
                                           em_iter=EM_ITER(n), cooc_epochs=params['cooc_epochs_param'],
                                           epochs=params['cooc_epochs_param']), scheduler=True, simple_model=simple_model)

        hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                        wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                        log_config={'metrics_after_convergence': True})
        densehmm = GaussianDenseHMM(n, mstep_config={'cooc_lr': cooc_lr_param, "l_uz": l_param, 'scheduler': em_scheduler,
                                                     'cooc_epochs': cooc_epochs_param, 'loss_type': 'square'},
                                    covariance_type='diag', em_iter=EM_ITER(n), logging_monitor=hmm_monitor,
                                    init_params="", params="stmc", early_stopping=True, opt_schemes={"cooc"},
                                    discrete_observables=m)

        densehmm.means_ = mu
        densehmm.fit_coocs(Y_true, lengths)
        return hmm_monitor.loss[-1]

    def callback(study, trial):
        if trial.value < 0.001:
            raise study.stop()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])
    params = study.best_params

    # provide data for main part  of the experiment
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
    wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                 "name": f"HMMlearn"})
    wandb_params["config"].update(dict(model="HMMlearn", m=0, l=0, lr=0,
                                       em_iter=EM_ITER(n), cooc_epochs=0,
                                       epochs=0), scheduler=False, simple_model=simple_model)

    hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                    wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                    log_config={'metrics_after_convergence': True})
    hmm_model = hmm.GaussianHMM(n, n_iter=EM_ITER(n))
    hmm_model.monitor_ = hmm_monitor
    hmm_model.fit(Y_true, lengths)

    preds = hmm_model.predict(Y_true, lengths)
    perm  = find_permutation(preds, X_true)
    hmm_monitor.run.finish()

    best_result["HMMlearn"] = {
        "time": time.perf_counter() - hmm_monitor._init_time,
        "logprob": hmm_model.score(Y_true, lengths),
        "acc": (X_true == np.array([perm[i] for i in preds])).mean(),
        "dtv_transmat": dtv(hmm_model.transmat_, A[perm, :][:, perm]),
        "dtv_startprob": dtv(hmm_model.startprob_, pi[perm]),
        "MAE_means": (abs(mu[perm] - hmm_model.means_[:, 0])).mean(),
        "MAE_sigma": (abs(sigma.reshape(-1)[perm] - hmm_model.covars_.reshape(-1))).mean()
    }

    # Dense HMM
    wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                 "name": f"dense--l={params['l_param']}-lr={params['cooc_lr_param']}-epochs={params['cooc_epochs_param']}"})
    wandb_params["config"].update(dict(model="dense_cooc", m=0, l=params['l_param'], lr=params['cooc_lr_param'],
                                       em_iter=EM_ITER(n), cooc_epochs=params['cooc_epochs_param'],
                                       epochs=params['cooc_epochs_param']), scheduler=True, simple_model=simple_model)

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
    hmm_monitor.run.finish()
    best_result["DenseCooc"] = {
        "time": time.perf_counter() - hmm_monitor._init_time,
        "logprob": densehmm.score(Y_true, lengths),
        "acc": (X_true == np.array([perm[i] for i in preds])).mean(),
        "dtv_transmat": dtv(densehmm.transmat_, A[perm, :][:, perm]),
        "dtv_startprob": dtv(densehmm.startprob_, pi[perm]),
        "MAE_means": (abs(mu[perm] - densehmm.means_[:, 0])).mean(),
        "MAE_sigma": (abs(sigma.reshape(-1)[perm] - densehmm.covars_.reshape(-1))).mean()
    }

    pca_z = PCA(n_components=2).fit(hmm_monitor.z[-1])
    z = [pca_z.transform(x) for x in hmm_monitor.z]

    z0 = hmm_monitor.z0

    pca_u = PCA(n_components=2).fit(np.transpose(hmm_monitor.u[-1]))
    u = [pca_u.transform(np.transpose(x)) for x in hmm_monitor.u]

    draw_embeddings(z, run, "z")
    draw_embeddings(z0, run, "z0")
    draw_embeddings(u, run, "u")

    run.finish()

    with open(f"{RESULT_DIR}/optuna_s{s}_T{T}_n{n}_simple_model{simple_model}.pkl",  "wb") as f:
        joblib.dump(study,  f)
    with open(f"{RESULT_DIR}/best_result_s{s}_T{T}_n{n}_simple_model{simple_model}.json",  "w") as f:
        json.dump(best_result,  f, indent=4)

    return 0


def run_true(dsize):
    return run_experiment(dsize, simple_model=True)

def run_false(dsize):
    return run_experiment(dsize, simple_model=False)

if __name__ == "__main__":
    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)
    wandb.require("service")

    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('dsize', metavar='N', type=int, nargs=3,
    #                     help='s, T, n')
    # dsize = parser.parse_args().dsize
    #
    # run_experiment(dsize, simple_model=True)
    # run_experiment(dsize, simple_model=False)

    # with mp.Pool(1) as pool:
    #     pool.map(run_true, data_sizes)
    #     pool.map(run_false, data_sizes)

    [(run_experiment(dsize, simple_model=True), run_experiment(dsize, simple_model=False)) for dsize in data_sizes]
