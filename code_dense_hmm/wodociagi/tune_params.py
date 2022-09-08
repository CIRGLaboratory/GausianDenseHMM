from hmmlearn import hmm
import numpy as np
import pandas as pd
import sys
from pathlib import Path
# sys.path.insert(0, Path(__file__).parent.absolute)
from models_gaussian_2d import GaussianDenseHMM, HMMLoggingMonitor, DenseHMMLoggingMonitor
import joblib
import json
from pathlib import Path
import optuna
from eval_utils import *
import tqdm
import argparse
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow.compat.v1 as tf
import time
from scipy.special import erf
tf.disable_v2_behavior()

np.random.seed(2022)

t = time.localtime()
RESULT_DIR = f'gaussian_dense_hmm_benchmark/eval-cooc-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'


models = dict(dense=GaussianDenseHMM, dense_em=GaussianDenseHMM)
monitors = dict(dense=DenseHMMLoggingMonitor, dense_em=DenseHMMLoggingMonitor)
algs = dict(dense="cooc", dense_em="em")


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("-s", type=int, help="number of sentences",  default=100)
    # parser.add_argument("-t", type=int, help="lengths of sentence", default=100)
    parser.add_argument("-n", type=int, help="number of states", default=10)
    parser.add_argument("-r", type=int, help="number of repetitions in each hyper-parameters evaluation", default=8)
    parser.add_argument("-q", type=int, help="number of hyper-parameters trials", default=64)
    # parser.add_argument("--simple-model", action="store_true", help="check simple example")
    parser.add_argument("-l", action="store_true", help="fix embedding length")
    # parser.add_argument("-i",  "--input", type=ascii, help="input data path", default="")
    parser.add_argument("-c",  "--covar-type", type=str, help="covariance type,  one of  diag, full, tied, spherical",
                        default='diag')
    args = parser.parse_args()
    return args.n, args.r, args.q, args.l, args.covar_type

def prepare_data():
    df_main = pd.read_excel('../../data/Dane_Uwr.xlsx', sheet_name='Surowe_hydraulika').ffill()
    df_main.columns = ['mtime', 'P1', 'V1', 'Q1']
    df_main = df_main.ffill()
    df_main.head()
    df_main["V_delta"] = np.array([0] + (df_main.V1[1:].values - df_main.V1[:-1].values).tolist())
    df_main.loc[(df_main.V_delta.abs() > 1e+3), "V_delta"] = 0
    seasonal_changes = df_main.V_delta.rolling(6 * 24 * 42, center=True, min_periods=2).mean().rolling(6 * 24 * 7, center=True, min_periods=2).mean()[df_main.mtime.dt.year == 2019]

    data = df_main.V_delta.rolling(24 * 6, center=True, min_periods=2).mean()[df_main.mtime.dt.year == 2019] - seasonal_changes
    lengths = np.array([24 * 7 * 6 for _ in range(data.shape[0] // (24 * 7 * 6))] + [
        data.shape[0] - (data.shape[0] // (24 * 7 * 6)) * 24 * 7 * 6])
    return data.values.reshape(-1, 1), lengths

def objective(trial, n, Y_true, lengths, covar_type, l=None, no_rep=8):
    # Init parameters
    if l is None:
        l_param = trial.suggest_int('l_param', n // 4, (n * 2) // 3)
    else:
        l_param = int(l)
    cooc_lr_param = trial.suggest_loguniform('cooc_lr_param', 1e-4, .75)
    cooc_epochs_param = trial.suggest_int('cooc_epochs_param', 1000, 1000000)
    lls = []

    def em_scheduler(max_lr, it):
        if it <= np.ceil(cooc_epochs_param / 3):
            return max_lr * np.cos(3 * (np.ceil(cooc_epochs_param / 3) - it) * np.pi * .33 / cooc_epochs_param)
        else:
            return max_lr * np.cos((it - np.ceil(cooc_epochs_param / 3)) * np.pi * .66 / cooc_epochs_param) ** 2

    # Check hyper-parameters
    for _ in range(no_rep):
        hmm_monitor = DenseHMMLoggingMonitor(tol=TOLERANCE, n_iter=0,
                              verbose=False, wandb_log=False)

        mstep_config = {'cooc_lr': cooc_lr_param, "l_uz": l_param, 'scheduler': em_scheduler,
                        'cooc_epochs': cooc_epochs_param, 'loss_type': 'square'}

        hmm_model = GaussianDenseHMM(n, mstep_config=mstep_config, verbose=False,
                          covariance_type=covar_type, em_iter=em_iter(n), logging_monitor=hmm_monitor,
                          init_params="stmc", params="stmc", early_stopping=True, opt_schemes={"cooc"})

        hmm_model.fit_coocs(Y_true, lengths)
        lls.append(hmm_model.score(Y_true, lengths))

    lls = np.array(lls)
    return lls.mean()


def tune_hyperparams(Y_true, lengths, n, covar_type):
    l = max(np.ceil(n / 3), 2) if l_fixed else None
    study = optuna.create_study(direction='maximize')

    study.optimize(
        lambda trial: objective(trial, n, Y_true, lengths, covar_type, l=int(l), no_rep=no_trials),
        n_trials=no_trials)

    with open(f"{RESULT_DIR}/optuna_cooc_n{n}_l{l_fixed}.pkl", "wb") as f:
        joblib.dump(study, f)

    best_params = study.best_params
    if l_fixed:
        best_params["l_param"] = l
    return best_params


def normal_cooc_prob(means, covars, nodes, A, covar_type):
    A_stationary = compute_stationary(A, False)

    B_scalars_tmp = None
    if means.shape[1] == 1:
        B_scalars_tmp = .5 * (
                1 + erf((nodes - np.transpose(means)) /
                np.transpose(covars[:, :, 0]) / np.sqrt(2)))

        B_scalars = np.transpose(B_scalars_tmp[1:, :] - B_scalars_tmp[:-1, :])

    elif means.shape[1] == 2:
        if covar_type == "full":
            mvn = tfp.distributions.MultivariateNormalTriL(means, covars)
            mvn_sample = mvn.sample(100000, seed=2022)
            B_scalars_tmp = tf.map_fn(
                lambda n: tf.reduce_mean(tf.cast(tf.reduce_all(mvn_sample <= n, axis=-1), mvn.dtype),
                                         axis=0), nodes)
        elif covar_type == "diag":
            B_scalars_tmp = tf.reduce_prod(.5 * (
                    1 + tf.erf(
                (tf.expand_dims(nodes, axis=-1) - tf.expand_dims(tf.transpose(means), axis=0)) / (
                        tf.nn.relu(tf.expand_dims(tf.transpose(covars), axis=0)) + 1e-10) / np.sqrt(2))), axis=1)

        B_scalars_tmp_wide = tf.reshape(B_scalars_tmp, np.array(nodes.shape) - 1)

        B_scalars = tf.transpose(tf.reshape(
            B_scalars_tmp_wide[:-1, 1:, :] - B_scalars_tmp_wide[:-1, :-1, :] - B_scalars_tmp_wide[1:, 1:,
                                                                               :] + B_scalars_tmp_wide[1:, :-1, :],
            (-1, means.shape[0])), name="B_scalars_cooc")
    else:
        raise Exception("Co-occurrences for dimensionality >=  3  not implemented.")

    theta = A * A_stationary[:, None]
    return np.matmul(np.transpose(B_scalars), np.matmul(theta, B_scalars))


def run_models(params, Y_true, lengths, no_rep, covar_type):
    best_result = dict()

    def em_scheduler(max_lr, it):
        if it <= np.ceil(params['cooc_epochs_param'] / 3):
            return max_lr * np.cos(3 * (np.ceil(params['cooc_epochs_param'] / 3) - it) * np.pi * .33 / params['cooc_epochs_param'])
        else:
            return max_lr * np.cos((it - np.ceil(params['cooc_epochs_param'] / 3)) * np.pi * .66 / params['cooc_epochs_param']) ** 2

    for _ in tqdm.tqdm(range(no_rep), desc=f"Evaluation"):

        wandb_params["init"].update({"job_type": "evaluate model",
                                     "name": "dense"})
        wandb_params["config"].update(
            dict(model="dense_cooc", l=int(params['l_param']), lr=params['cooc_lr_param'],
                 cooc_epochs=params['cooc_epochs_param'], scheduler=True, covar_type=covar_type))

        hmm_monitor = DenseHMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                              wandb_log=True, wandb_params=wandb_params, true_vals=None,
                              log_config={'metrics_after_convergence': True})
        densehmm = GaussianDenseHMM(n, mstep_config={'cooc_epochs': params['cooc_epochs_param'],
                                          'cooc_lr': params['cooc_lr_param'],
                                          "l_uz": int(params['l_param']),
                                          'loss_type': 'square',
                                          'scheduler': em_scheduler},
                         covariance_type=covar_type, logging_monitor=hmm_monitor,
                         init_params="", params="stmc", early_stopping=False, opt_schemes={"cooc"})
        densehmm.fit_coocs(Y_true, lengths)

        best_result["dense"].append(
            {
                "time": time.perf_counter() - hmm_monitor._init_time,
                "logprob": densehmm.score(Y_true, lengths),
                "dtv_omega": dtv(empirical_cooc_prob(densehmm._to_discrete(Y_true), n + 2, lengths),
                                 normal_cooc_prob(densehmm.means_.reshape(-1), densehmm.covars_.reshape(-1),
                                                  densehmm.discrete_nodes, densehmm.transmat_, covar_type))
            }
        )

        wandb_params["init"].update({"job_type": f"evaluate model",
                                     "name": f"HMMlearn"})
        wandb_params["config"].update(dict(model="HMMlearn", l=0, lr=0,
                                           em_iter=em_iter(n), cooc_epochs=0,
                                           epochs=0), scheduler=False, simple_model=simple_model)

        hmm_monitor = HMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                        wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                        log_config={'metrics_after_convergence': True})
        hmm_model = hmm.GaussianHMM(n, n_iter=em_iter(n), covariance_type=covar_type)
        hmm_model.monitor_ = hmm_monitor
        hmm_model.fit(Y_true, lengths)

        best_result["hmmlearn"].append(
            {
                "time": time.perf_counter() - hmm_monitor._init_time,
                "logprob": hmm_model.score(Y_true, lengths),
                "dtv_omega": dtv(empirical_cooc_prob(densehmm._to_discrete(Y_true), n + 2, lengths),
                                 normal_cooc_prob(hmm_model.means_.reshape(-1), hmm_model.covars_.reshape(-1),
                                                  hmm_model.discrete_nodes, hmm_model.transmat_, covar_type))
            }
        )

    with open(f"{RESULT_DIR}/best_result_n{n}_l{l_fixed}.json", "w") as f:
        json.dump(best_result, f, indent=4)
    return 0


if __name__ == "__main__":
    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)
    n, no_reps, no_trials, l_fixed, covar_type = parse_args()

    Y_true, lengths = prepare_data()

    params = tune_hyperparams(Y_true, lengths, n, covar_type)
    run_models(params, Y_true, lengths, no_reps, covar_type)
