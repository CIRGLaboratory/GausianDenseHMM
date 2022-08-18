import wandb
import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from celluloid import Camera

from ssm.util import find_permutation
from utils import dtv, permute_embeddings, compute_stationary, empirical_coocs
from scipy.special import erf

simple_model_params = {"mu": 10, "sigma": 1}
complicated_model_params = {"mu": 5, "sigma": 2}


def em_iter(n):
    return 10 * n


TOLERANCE = 5e-5
N_TRIALS = 32  # OPTUNA
t = time.localtime()


def prepare_params(n, simple_model=True):
    A = np.exp(np.random.uniform(0, 5, size=(n, n)))
    A /= A.sum(axis=1)[:, np.newaxis]

    pi = compute_stationary(A)

    if simple_model:
        mu = np.arange(n) * simple_model_params["mu"]
        sigma = np.ones(shape=n) * simple_model_params["sigma"]
    else:
        mu = np.random.uniform(0, n * complicated_model_params["mu"], size=n)
        sigma = np.random.uniform(0.001,  1, size=n) * complicated_model_params["sigma"]
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

    EM_ITER_tmp = em_iter(n)

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
            "group": f"eval-cooc-{t.tm_year}-{t.tm_mon}-{t.tm_mday}",
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

    return s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, data, em_scheduler


def draw_embeddings(z, run=None, name="?"):
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
    if run is not None:
        run.log({f"Embaddings trajectory:  {name}": wandb.Html(animation.to_html5_video())})
    plt.close()


def objective(trial, n, m, model, monitor,  Y_true, lengths, mu, em_scheduler, alg="coooc", l=None):
    # Init pparameters
    if l is None:
        l_param = trial.suggest_int('l_param', n // 4, n // 2)
    else:
        l_param = int(l)
    cooc_lr_param = trial.suggest_loguniform('cooc_lr_param', 1e-4, .5)
    cooc_epochs_param = trial.suggest_int('cooc_epochs_param', 10000, 100000)
    lls = []

    # Check hyper-parameters
    for _ in range(8):
        hmm_monitor = monitor(tol=TOLERANCE, n_iter=0,
                              verbose=False, wandb_log=False, log_config={'metrics_after_convergence': True})
        if alg == "cooc":
            mstep_config = {'cooc_lr': cooc_lr_param, "l_uz": l_param, 'scheduler': em_scheduler,
                            'cooc_epochs': cooc_epochs_param, 'loss_type': 'square'}
        else:
            mstep_config = {'em_lr': cooc_lr_param, "l_uz": l_param, 'scheduler': em_scheduler,
                            'em_epochs': cooc_epochs_param // 5000}
        hmm_model = model(n, mstep_config=mstep_config, verbose=False,
                          covariance_type='diag', em_iter=em_iter(n), logging_monitor=hmm_monitor,
                          init_params="", params="stmc", early_stopping=True, opt_schemes={"cooc"},
                          discrete_observables=m)

        hmm_model.means_ = mu

        if alg == "cooc":
            hmm_model.fit_coocs(Y_true, lengths)
        elif alg == "em":
            hmm_model.fit(Y_true, lengths)
        else:
            raise ValueError("Unknown learning algorithm.  Must be one of: cooc, em.")

        lls.append(hmm_model.score(Y_true, lengths))

    lls = np.array(lls)
    # optimize for log-likelihood and stability of the solution (no ground truth needed)
    return lls.mean(), lls.std()

def empirical_cooc_prob(Xd, m, lengths):
    freqs, gt_omega_emp = empirical_coocs(Xd, m, lengths=lengths)
    return np.reshape(gt_omega_emp, newshape=(m, m))

def normal_cooc_prob(means, covars, Qs, A):
    A_stationary = compute_stationary(A, False)
    B_scalars_tmp = .5 * (1 + erf((Qs[:-1, np.newaxis] - np.transpose(means)) / np.transpose(covars) / np.sqrt(2)))
    B_scalars_tmp = np.concatenate([np.zeros((1, B_scalars_tmp.shape[1])), B_scalars_tmp, np.ones((1, B_scalars_tmp.shape[1]))], axis=0)
    B_scalars = np.transpose(B_scalars_tmp[1:, :] - B_scalars_tmp[:-1, :])
    theta = A * A_stationary[:, None]
    return np.matmul(np.transpose(B_scalars), np.matmul(theta, B_scalars))
