from hmmlearn import hmm
from models_gaussian import GaussianDenseHMM, HMMLoggingMonitor, DenseHMMLoggingMonitor
from models_gaussian_A import GaussianDenseHMM as CoocHMM
import joblib
import json
from pathlib import Path
import optuna
import multiprocessing as mp
from eval_utils import *
import tqdm

np.random.seed(2022)

t = time.localtime()
RESULT_DIR = f'gaussian_dense_hmm_benchmark/eval-cooc-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'

data_sizes = [  # (s, T, n)
    (1000, 10000, 100),
    (100, 1000, 8)
]


def run_experiment(dsize, simple_model=True, l_fixed=True):
    ## setup

    best_result = {}
    s, T, n, pi, A, mu, sigma, result, true_values, wandb_params, X_true, Y_true, lengths, _, em_scheduler = init_experiment(dsize, simple_model)
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (mu[1:] + mu[:-1]) / 2,
                            np.array([Y_true.max(), np.infty])])
    m = nodes.shape[0] - 1

    models = dict(cooc=CoocHMM, dense=GaussianDenseHMM, dense_em=GaussianDenseHMM)
    monitors = dict(cooc=DenseHMMLoggingMonitor, dense=DenseHMMLoggingMonitor, dense_em=DenseHMMLoggingMonitor)
    algs = dict(cooc="cooc", dense="cooc", dense_em="em")

    ## Tune hyper-parameters
    l = np.ceil(n / 3) if l_fixed else None
    best_params = dict()
    for name in tqdm.tqdm(["cooc", "dense", "dense_em"],  desc="Hyper-param tuning"):
        study = optuna.create_study(directions=['maximize', 'minimize'])
        study.optimize(
            lambda trial: objective(trial, n, m, models[name], monitors[name], Y_true, lengths, mu, em_scheduler,
                                    alg=algs[name], l=int(l)), n_trials=N_TRIALS)
        with open(f"{RESULT_DIR}/optuna_{name}_s{s}_T{T}_n{n}_simple_model{simple_model}_l{l_fixed}.pkl", "wb") as f:
            joblib.dump(study, f)
        best_params[name] = study.best_trials[0].params
        if l_fixed:
            best_params[name]["l_param"] = l


    ## Evaluate models
    #  prepare  new data
    data = [my_hmm_sampler(pi, A, mu, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    lengths = [len(x[1]) for x in data]
    nodes_tmp = mu
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (nodes_tmp[1:] + nodes_tmp[:-1]).reshape(-1) / 2,
                            np.array([Y_true.max(), np.infty])])

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

    Y_disc = (Y_true > nodes.reshape(1, -1)).sum(axis=-1).reshape(-1, 1)

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
                "MAE_sigma": (abs(sigma.reshape(-1)[perm] - hmm_model.covars_.reshape(-1))).mean(),
                "dtv_omega": dtv(empirical_cooc_prob(Y_disc, n, lengths),
                                 normal_cooc_prob(hmm_model.means_.reshape(-1), hmm_model.covars_.reshape(-1), nodes, A))
            }
        )

    # Custom models
    for name in tqdm.tqdm(["cooc", "dense", "dense_em"], desc="Model building"):
        model = models[name]
        monitor = monitors[name]
        alg = algs[name]
        params = best_params[name]
        best_result[name] = list()
        wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                     "name": f"name-l={params['l_param']}-lr={params['cooc_lr_param']}-epochs={params['cooc_epochs_param']}"})
        wandb_params["config"].update(
            dict(model="dense_cooc", m=0, l=int(params['l_param']), lr=params['cooc_lr_param'],
                 em_iter=em_iter(n), cooc_epochs=params['cooc_epochs_param'],
                 epochs=params['cooc_epochs_param']), scheduler=True,
            simple_model=simple_model)

        for _ in tqdm.tqdm(range(10), desc=f"Training {name}"):
            hmm_monitor = monitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                  wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                  log_config={'metrics_after_convergence': True})
            # kmeans = KMeans(n_clusters=n, random_state=0).fit(Y_true)
            densehmm = model(n, mstep_config={'cooc_epochs': params['cooc_epochs_param'],
                                              'cooc_lr': params['cooc_lr_param'],
                                              "l_uz": int(params['l_param']),
                                              'loss_type': 'square',
                                              'scheduler': em_scheduler},
                             covariance_type='diag', logging_monitor=hmm_monitor, nodes=nodes,
                             init_params="", params="stmc", early_stopping=False, opt_schemes={"cooc"},
                             discrete_observables=m)
            if alg == "cooc":
                densehmm.fit_coocs(Y_true, lengths)
            else:
                densehmm.fit(Y_true, lengths)

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
                    "MAE_sigma": (abs(sigma.reshape(-1)[perm] - densehmm.covars_.reshape(-1))).mean(),
                    "dtv_omega": dtv(empirical_cooc_prob(Y_disc, n, lengths),
                                     normal_cooc_prob(densehmm.means_.reshape(-1), densehmm.covars_.reshape(-1), nodes, A))
                }
            )

    with open(f"{RESULT_DIR}/best_result_s{s}_T{T}_n{n}_simple_model{simple_model}_l{l_fixed}.json", "w") as f:
        json.dump(best_result, f, indent=4)
    return 0


def run_true(dsize):
    return run_experiment(dsize, simple_model=True, l_fixed=True)


def run_false(dsize):
    return run_experiment(dsize, simple_model=True, l_fixed=False)


if __name__ == "__main__":
    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)

    with mp.Pool(8) as pool:
        pool.map(run_true, data_sizes)
        pool.map(run_false, data_sizes)
