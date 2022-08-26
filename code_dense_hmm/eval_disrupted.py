import argparse
from hmmlearn import hmm
from models_gaussian import GaussianDenseHMM, HMMLoggingMonitor, DenseHMMLoggingMonitor
import joblib
import json
from pathlib import Path
import optuna
import multiprocessing as mp
from eval_utils import *
import tqdm


np.random.seed(2022)

t = time.localtime()
RESULT_DIR = f'gaussian_dense_hmm_benchmark/eval-disrupted-fixed-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'

data_sizes = [  # (s, T, n)
    (100, 200, 3)
]

no_rep = 8

# From previous hyperparameter tuning (eval_cooc.py)
l_param = 2
lr_param = 0.036192647318347725
epochs_param = 481442

def generate_data_1(pi, A, mu, sigma, T, s, param=0.5):
    delta_mu = mu[1:] - mu[:-1]
    n2 = mu.shape[0] // 2
    mu_tmp = mu.copy()
    mu_tmp[1:n2] = mu_tmp[1:n2] - delta_mu[:(n2-1)] * param
    mu_tmp[n2:] = mu_tmp[n2:] - delta_mu[(n2-1):] * param
    data = [my_hmm_sampler(pi, A, mu_tmp, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    lengths = np.array([len(x[1]) for x in data])
    nodes_tmp = np.array([Y_true[X_true == i].mean() for i in range(A.shape[0])])
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (nodes_tmp[1:] + nodes_tmp[:-1]).reshape(-1) / 2,
                            np.array([Y_true.max(), np.infty])])
    return X_true, Y_true, lengths, nodes


def generate_data_2(pi, A, mu, sigma, T, s, param=0.5):
    delta_mu = ((mu[1:] - mu[:-1]) * param).cumsum()
    mu_tmp = mu.copy()
    mu_tmp[1:] = mu_tmp[1:] - delta_mu
    data = [my_hmm_sampler(pi, A, mu_tmp, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    lengths = np.array([len(x[1]) for x in data])
    nodes_tmp = np.array([Y_true[X_true == i].mean() for i in range(A.shape[0])])
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (nodes_tmp[1:] + nodes_tmp[:-1]).reshape(-1) / 2,
                            np.array([Y_true.max(), np.infty])])
    return X_true, Y_true, lengths, nodes


def generate_data_3(pi, A, mu, sigma, T, s, param=0.25):
    data = [my_hmm_sampler(pi, A, mu, sigma, T) for _ in range(s)]
    X_true = np.concatenate([np.concatenate(y[0]) for y in data])  # states
    Y_true = np.concatenate([x[1] for x in data])  # observations
    maxi = (mu[1:] - mu[:-1]).max() * param
    Y_true += np.array([[i * maxi / (s*T)] for i in range(s*T)])
    lengths = np.array([len(x[1]) for x in data])
    nodes_tmp = np.array([Y_true[X_true == i].mean() for i in range(A.shape[0])])
    nodes = np.concatenate([np.array([-np.infty, Y_true.min()]),
                            (nodes_tmp[1:] + nodes_tmp[:-1]).reshape(-1) / 2,
                            np.array([Y_true.max(), np.infty])])
    return X_true, Y_true, lengths, nodes


disruptions = [generate_data_1, generate_data_2, generate_data_3]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--disruption", type=int, help="number of disruption 1, ..., 6",
                        choices=[1, 2, 3])
    parser.add_argument("-p", "--param", type=float, help="param of disruption")
    parser.add_argument("-s", "--simple", type=int, help="simple (separable) distributions", default=1)
    args = parser.parse_args()
    return args.disruption,  args.param,  bool(args.simple)


def run_experiment(dsize, disruption, param, simple_model=True, l_fixed=True):
    s = dsize[0]
    T = dsize[1]
    n = dsize[2]
    pi, A, mu, sigma = prepare_params(n, simple_model)
    best_result = dict()
    X_true, Y_true, lengths, nodes = disruptions[disruption - 1](pi, A, mu, sigma, T, s, param=param)

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
            "group": f"eval-disrupted-cooc-final-{t.tm_year}-{t.tm_mon}-{t.tm_mday}",
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
    m = nodes.shape[0] - 1

    models = dict(dense=GaussianDenseHMM)
    monitors = dict(dense=DenseHMMLoggingMonitor)
    algs = dict(dense="cooc")

    ## Tune hyper-parameters
    l = max(np.ceil(n / 3),  2) if l_fixed else None
    best_params = dict()
    name = "dense"


    # HMMlearn
    best_result["HMMlearn"] = list()
    wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                 "name":  f"HMMlearn-dis{disruption}-p{param}"})
    wandb_params["config"].update(dict(model="HMMlearn", m=0, l=0, lr=0,
                                       em_iter=em_iter(n), cooc_epochs=0,
                                       epochs=0), scheduler=False, simple_model=simple_model)

    Y_disc = (Y_true > nodes.reshape(1, -1)).sum(axis=-1).reshape(-1, 1)

    for _ in range(no_rep):
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
                "dtv_omega": dtv(empirical_cooc_prob(Y_disc, n+2, lengths),
                                 normal_cooc_prob(hmm_model.means_.reshape(-1), hmm_model.covars_.reshape(-1), nodes[1:], A))
            }
        )

    # Custom models
    model = models[name]
    monitor = monitors[name]
    alg = algs[name]
    params = dict(l_param=int(l_param), cooc_lr_param=lr_param, cooc_epochs_param=epochs_param)
    best_result[name] = list()
    wandb_params["init"].update({"job_type": f"n={n}-s={s}-T={s}-simple={simple_model}",
                                 "name": f"dis{disruption}-p{param}-l={params['l_param']}-lr={params['cooc_lr_param']}-epochs={params['cooc_epochs_param']}"})
    wandb_params["config"].update(
        dict(model="dense_cooc", m=0, l=int(params['l_param']), lr=params['cooc_lr_param'],
             em_iter=em_iter(n), cooc_epochs=params['cooc_epochs_param'],
             epochs=params['cooc_epochs_param']), scheduler=True,
        simple_model=simple_model)

    for _ in tqdm.tqdm(range(no_rep), desc=f"Training {name}"):
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
                "dtv_omega": dtv(empirical_cooc_prob(Y_disc, n+2, lengths),
                                 normal_cooc_prob(densehmm.means_.reshape(-1), densehmm.covars_.reshape(-1), nodes[1:], A))
            }
        )

    with open(f"{RESULT_DIR}/best_result_d{disruption}_p{param}_s{s}_T{T}_n{n}_simple_model{simple_model}_l{l_fixed}.json", "w") as f:
        json.dump(best_result, f, indent=4)
    return 0


if __name__ == "__main__":
    disruption, param, simple_model = parse_args()

    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)

    run_experiment(data_sizes[0], disruption, param, simple_model=simple_model, l_fixed=True)


