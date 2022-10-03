import collections
import pickle

import pandas as pd
import numpy as np
import time
import json
import holidays
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from models_gaussian_2d import *


n = [4, 6, 10]
l = [2, 3, 4]
lr = [0.01, 0.05, 0.10, 0.20]
# lr = [0.05, 0.10, 0.20]
k = [3, 5, 13, 20]
TOLERANCE = 1e-4

interval = 24 * 7 * 6
t = time.localtime()
month_len = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
time_range_d = pd.date_range("00:00:00", "23:50:00", 24*6)
weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "holiday"]


def provide_data():
    df_main = pd.read_excel('../../data/Dane_Uwr.xlsx', sheet_name='Surowe_hydraulika').ffill()
    df_main.columns = ['mtime', 'P1', 'V1', 'Q1']
    df_main = df_main.ffill()
    df_main["V_delta"] = (np.array([0] + (df_main.V1[1:].values - df_main.V1[:-1].values).tolist()))
    df_main.loc[(df_main.V_delta.abs() > 50), "V_delta"] = 50 # df_main.loc[(df_main.V_delta.abs() < 50), "V_delta"].mean()
    df_main["V_delta"] = df_main["V_delta"].rolling(6, center=True, min_periods=2).mean()

    time_range = pd.DataFrame({"mtime":pd.date_range(df_main.loc[df_main.mtime.dt.year == 2018, :].mtime.min(), df_main.loc[(df_main.mtime.dt.year == 2019), :].mtime.max(), 24*6*(365+365))})
    df_main = pd.merge(time_range, df_main, on="mtime", how="left")
    df_main["V_delta"] = (df_main["V_delta"].bfill().ffill() + df_main["V_delta"].ffill().bfill()) / 2

    df_main["V_delta_der"] = np.concatenate([np.array([0]), np.array([df_main["V_delta"].values[1] - df_main["V_delta"].values[0]]),
                                             df_main["V_delta"].values[2:] - df_main["V_delta"].values[:-2] / 2 - df_main["V_delta"].values[1:-1] / 2])

    seasonal_changes = df_main.V_delta.rolling(interval * 6, center=True, min_periods=2).mean() \
        .rolling(interval, center=True, min_periods=2).mean()

    # data = df_main.loc[:, ['V_delta', 'V_delta_der']]
    data = df_main.loc[:, ['V_delta']]
    data.V_delta = data.V_delta - seasonal_changes

    data_train = data[(df_main.mtime.dt.year == 2018)]
    data_test  = data[(df_main.mtime.dt.year == 2019)]

    lengths = np.array([interval for _ in range(data_train.shape[0] // (interval))] + [
        data_train.shape[0] - (data_train.shape[0] // (interval)) * interval])
    Y_true = data_train.values.reshape(-1, 1)

    lengths_test = np.array([interval for _ in range(data_test.shape[0] // (interval))] + [
        data_test.shape[0] - (data_test.shape[0] // (interval)) * interval])
    Y_test = data_test.values.reshape(-1, 1)

    target = pd.Series(pd.date_range(df_main.loc[df_main.mtime.dt.year == 2018, :].mtime.dt.floor('d').min(),
                                     df_main.loc[df_main.mtime.dt.year == 2018, :].mtime.dt.floor('d').max(),
                                     365)).apply(lambda d: (d in holidays.PL()) | (d.weekday() > 4))

    target_test = pd.Series(pd.date_range(df_main.loc[(df_main.mtime.dt.year == 2019), :].mtime.dt.floor('d').min(),
                                          df_main.loc[(df_main.mtime.dt.year == 2019), :].mtime.dt.floor('d').max(),
                                          365)).apply(lambda d: (d in holidays.PL()) | (d.weekday() > 4))

    target_w = pd.Series(pd.date_range(df_main.loc[df_main.mtime.dt.year == 2018, :].mtime.dt.floor('d').min(),
                                       df_main.loc[df_main.mtime.dt.year == 2018, :].mtime.dt.floor('d').max(),
                                       365)).apply(
        lambda d: 7 if d in holidays.PL() else d.weekday())

    target_w_test = pd.Series(pd.date_range(df_main.loc[(df_main.mtime.dt.year == 2019), :].mtime.dt.floor('d').min(),
                                            df_main.loc[(df_main.mtime.dt.year == 2019), :].mtime.dt.floor('d').max(),
                                            365)).apply(
        lambda d: 7 if d in holidays.PL() else d.weekday())

    return Y_true, lengths, Y_test, lengths_test, target, target_test, target_w, target_w_test


def provide_logs():
    true_values = None
    wandb_params = {
        "init": {
            "project": "gaussian-dense-hmm-wodociagi",
            "entity": "cirglaboratory",
            "save_code": True,
            "group": f"models-eval-2018",
            "job_type": f"{t.tm_year}-{t.tm_mon}-{t.tm_mday}",
            "name": f"test3",
            "reinit": True
        },
        "config": {
            "n": 0,
            "s": 52,
            "T": 24*6*7,
            "model": None,
            "m": None,
            "l": 0,
            "lr": 0,
            "em_epochs": None,
            "em_iter": None,
            "cooc_epochs": None,
            "simple_model": None
        }
    }

    mstep_cofig = {"cooc_lr": 0, "cooc_epochs": 0, "l_uz": 0,
                   'loss_type': 'square', "scheduler": None}

    return wandb_params, mstep_cofig, true_values

def build_model(eta, n_, l_, Y_true, lengths, Y_test, lengths_test, wandb_params, mstep_cofig, true_values):
    ITER = 50000 + n_ * 10000
    def em_scheduler(max_lr, it):
        if it <= np.ceil(2 * ITER / 3):
            return max_lr * np.cos((np.ceil(ITER * 2 / 3) - it / 2) / ITER * np.pi * .67)
        else:
            return max_lr * np.cos(3 * (np.ceil(ITER * 2 / 3) - it) * np.pi * .33 / ITER) ** 3

    # update configs
    wandb_params['config'].update({"lr": eta, "l": l_, "n": n_, "cooc_epochs": ITER})
    mstep_cofig.update({"cooc_lr": eta, "l_uz": l_, "scheduler": em_scheduler})
    # train
    hmm_monitor = DenseHMMLoggingMonitor(tol=TOLERANCE, n_iter=0, verbose=True,
                                         wandb_log=True, wandb_params=wandb_params, true_vals=true_values,
                                         log_config={'metrics_after_convergence': True})
    densehmm = GaussianDenseHMM(n_, mstep_config=mstep_cofig,
                                covariance_type='diag', opt_schemes={"cooc"},
                                logging_monitor=hmm_monitor,
                                init_params="stmc", params="stmc", early_stopping=True)

    start = time.perf_counter()
    densehmm.fit_coocs(Y_true, lengths)
    done_in = time.perf_counter() - start

    # save model
    u, z, z0 = densehmm.get_representations()

    loss = list(hmm_monitor.omega_dtv)
    score = densehmm.score(Y_true, lengths)

    # provide predictions
    states = densehmm.predict(Y_true, lengths).reshape(1, -1)
    # provide predictions
    states_test = densehmm.predict(Y_test, lengths_test).reshape(1, -1)

    model_log = {
        'means': densehmm.means_,
        'covars': densehmm.covars_,
        'transmat': densehmm.transmat_,
        'startprob': densehmm.startprob_,
        'u': u,
        'z': z,
        'z0': z0,
        'z_traj': list(hmm_monitor.z),
        'z0_traj': list(hmm_monitor.z0),
        'u_traj': list(hmm_monitor.u),
        'omega_dtv': loss,
        'log-lik': score,
        'eta': eta,
        'l': l_,
        'n': n_,
        'iter': ITER,
        'date': t,
        'training_time': done_in
    }

    model_log2 = {
        'means': densehmm.means_.tolist(),
        'covars': densehmm.covars_.tolist(),
        'transmat': densehmm.transmat_.tolist(),
        'startprob': densehmm.startprob_.tolist(),
        'u': u.tolist(),
        'z': z.tolist(),
        'z0': z0.tolist(),
        'z_traj': list(hmm_monitor.z),
        'z0_traj': list(hmm_monitor.z0),
        'u_traj': list(hmm_monitor.u),
        'omega_dtv': loss,
        'log-lik': score,
        'eta': eta,
        'l': l_,
        'n': n_,
        'iter': ITER,
        'date': t,
        'training_time': done_in
    }
    return densehmm, states, states_test, loss, score, model_log, model_log2


def acc_perm(a):
    return max(a, 1 - a)

def cluster_daily(model_log):
    u_fin, z_fin, z0_fin = model_log['u'], model_log['z'], model_log['z0']
    uz_fin = np.concatenate([u_fin, np.transpose(z_fin)], axis=1)

    kmeans = KMeans(n_clusters=2).fit(uz_fin)
    uz_label = kmeans.labels_

    daily_bin = uz_label[states].reshape(-1, 24*6)
    daily_bin_test = uz_label[states_test].reshape(-1, 24*6)

    acc_final = 0
    acc_test = 0
    c1_final = 0
    c2_final = 24*6

    for c1 in range(24*6):
        for c2 in range(c1+1, 24*6):
            acc_tmp =  ((daily_bin[:, :c1] == 0).sum() + (daily_bin[:, c1:c2] == 1).sum() + (daily_bin[:, c2:] == 0).sum()) / (365 * 24 * 6)
            if acc_tmp > acc_final:
                acc_final = acc_tmp
                acc_test = ((daily_bin_test[:, :c1] == 0).sum() + (daily_bin_test[:, c1:c2] == 1).sum() + (daily_bin_test[:, c2:] == 0).sum()) / (365 * 24 * 6)
                c1_final = c1
                c2_final = c2

    result = {
        "train_ACC": acc_final,
        "test_ACC": acc_test,
        "timestamp_1":  str(time_range_d.values[c1_final]),
        "timestamp_2": str(time_range_d.values[c2_final])
    }

    return result


def cluster_weekly(model_log, Y_true, Y_test, states, states_test, target, target_test):
    u_fin, z_fin, z0_fin = model_log['u'], model_log['z'], model_log['z0']
    n_ = model_log["n"]
    uz_fin = np.concatenate([u_fin, np.transpose(z_fin)], axis=1)

    kmeans_Y = KMeans(n_clusters=2).fit(Y_true.reshape(-1, 24*6*2))
    kmeans_s = KMeans(n_clusters=2).fit(np.identity(n_)[states].reshape((365, -1)))
    kmeans_e = KMeans(n_clusters=2).fit(uz_fin[states.reshape(-1), :].reshape((365, -1)))

    result = {
        "water_demand": {
            "train_ACC": acc_perm((kmeans_Y.labels_ == target).mean()),
            "test_ACC": acc_perm((kmeans_Y.predict(Y_test.reshape(-1, 24*6*2)) == target_test).mean())
        },
        "state_IDs": {
            "train_ACC": acc_perm((kmeans_s.labels_ == target).mean()),
            "test_ACC": acc_perm((kmeans_s.predict(np.identity(n_)[states_test].reshape((365, -1))) == target_test).mean()),
        },
        "embedding": {
            "train_ACC": acc_perm((kmeans_e.labels_ == target).mean()),
            "test_ACC": acc_perm((kmeans_e.predict(uz_fin[states_test.reshape(-1), :].reshape((365, -1))) == target_test).mean()),
        }
    }

    return result


def present_weekday(knn_preds, knn_preds_test, target, target_test):
    result = {
        "train_ACC": (target == knn_preds).mean(),
        "test_ACC": (target_test == knn_preds_test).mean(),
        "train_PRECISION": precision_score(target, knn_preds),
        "test_PRECISION": precision_score(target_test, knn_preds_test),
        "train_RECALL": recall_score(target, knn_preds),
        "test_RECALL": recall_score(target_test, knn_preds_test),
        "train_CONFMAT":  confusion_matrix(target, knn_preds).tolist(),
        "test_CONFMAT": confusion_matrix(target_test, knn_preds_test).tolist(),
        "train_MONTH_ACC": [(knn_preds == target)[month_len[i - 1]:month_len[i]].mean() for i in range(1, month_len.shape[0])],
        "test_MONTH_ACC": [(knn_preds_test == target_test)[month_len[i - 1]:month_len[i]].mean() for i in range(1, month_len.shape[0])]
    }
    return result


def classify_working(k, model_log, Y_true, Y_test, states, states_test, target, target_test):
    u_fin, z_fin, z0_fin = model_log['u'], model_log['z'], model_log['z0']
    n_ = model_log["n"]
    uz_fin = np.concatenate([u_fin, np.transpose(z_fin)], axis=1)

    knn_Y = KNeighborsClassifier(n_neighbors=k).fit(Y_true.reshape(-1, 24 * 6 * 2), target)
    knn_s = KNeighborsClassifier(n_neighbors=k).fit(np.identity(n_)[states].reshape((365, -1)), target)
    knn_e = KNeighborsClassifier(n_neighbors=k).fit(uz_fin[states.reshape(-1), :].reshape((365, -1)), target)

    knn_preds_Y = knn_Y.predict(Y_true.reshape(-1, 24 * 6 * 2))
    knn_preds_Y_test = knn_Y.predict(Y_test.reshape(-1, 24 * 6 * 2))

    knn_preds_s = knn_s.predict(np.identity(n_)[states].reshape((365, -1)))
    knn_preds_s_test = knn_s.predict(np.identity(n_)[states_test].reshape((365, -1)))

    knn_preds_e = knn_e.predict(uz_fin[states.reshape(-1), :].reshape((365, -1)))
    knn_preds_e_test = knn_e.predict(uz_fin[states_test.reshape(-1), :].reshape((365, -1)))

    result = {
        "water_demand": present_weekday(knn_preds_Y, knn_preds_Y_test, target, target_test),
        "state_IDs": present_weekday(knn_preds_s, knn_preds_s_test, target, target_test),
        "embedding": present_weekday(knn_preds_e, knn_preds_e_test, target, target_test)
    }
    return result


def classify_weekday(k, model_log, Y_true, Y_test, states, states_test, target_w, target_w_test):
    u_fin, z_fin, z0_fin = model_log['u'], model_log['z'], model_log['z0']
    n_ = model_log["n"]
    uz_fin = np.concatenate([u_fin, np.transpose(z_fin)], axis=1)

    knn_Y = KNeighborsClassifier(n_neighbors=k).fit(Y_true.reshape(-1, 24 * 6 * 2), target_w)
    knn_s = KNeighborsClassifier(n_neighbors=k).fit(np.identity(n_)[states].reshape((365, -1)), target_w)
    knn_e = KNeighborsClassifier(n_neighbors=k).fit(uz_fin[states.reshape(-1), :].reshape((365, -1)), target_w)

    knn_preds_Y = knn_Y.predict(Y_true.reshape(-1, 24 * 6 * 2))
    knn_preds_Y_test = knn_Y.predict(Y_test.reshape(-1, 24 * 6 * 2))

    knn_preds_s = knn_s.predict(np.identity(n_)[states].reshape((365, -1)))
    knn_preds_s_test =knn_s.predict(np.identity(n_)[states_test].reshape((365, -1)))

    knn_preds_e = knn_e.predict(uz_fin[states.reshape(-1), :].reshape((365, -1)))
    knn_preds_e_test = knn_e.predict(uz_fin[states_test.reshape(-1), :].reshape((365, -1)))

    result = {
        "water_demand": {
            "train_ACC": (knn_preds_Y == target_w).mean(),
            "test_ACC": (knn_preds_Y_test == target_w_test).mean(),
            "train_MONTH_ACC": [(knn_preds_Y == target_w)[month_len[i - 1]:month_len[i]].mean() for i in
                                range(1, month_len.shape[0])],
            "test_MONTH_ACC": [(knn_preds_Y_test == target_w_test)[month_len[i - 1]:month_len[i]].mean() for i in
                               range(1, month_len.shape[0])],
            "train_CONFMAT": confusion_matrix(target_w, knn_preds_Y).tolist(),
            "test_CONFMAT": confusion_matrix(target_w_test, knn_preds_Y_test).tolist(),
            **{weekdays[i]: present_weekday(knn_preds_Y == i, knn_preds_Y_test == i, target_w == i, target_w_test == i) for i in range(8)},
            "working_day": present_weekday(knn_preds_Y < 5, knn_preds_Y_test < 5, target_w < 5, target_w_test < 5)
        },
        "state_IDs": {
            "train_ACC": (knn_preds_s == target_w).mean(),
            "test_ACC": (knn_preds_s_test == target_w_test).mean(),
            "train_MONTH_ACC": [(knn_preds_s == target_w)[month_len[i - 1]:month_len[i]].mean() for i in
                                range(1, month_len.shape[0])],
            "test_MONTH_ACC": [(knn_preds_s_test == target_w_test)[month_len[i - 1]:month_len[i]].mean() for i in
                               range(1, month_len.shape[0])],
            "train_CONFMAT": confusion_matrix(target_w, knn_preds_s).tolist(),
            "test_CONFMAT": confusion_matrix(target_w_test, knn_preds_s_test).tolist(),
            **{weekdays[i]: present_weekday(knn_preds_s == i, knn_preds_s_test == i, target_w == i, target_w_test == i) for i in range(8)},
            "working_day": present_weekday(knn_preds_s < 5, knn_preds_s_test < 5, target_w < 5, target_w_test < 5)
        },
        "embedding": {
            "train_ACC": (knn_preds_e == target_w).mean(),
            "test_ACC": (knn_preds_e_test == target_w_test).mean(),
            "train_MONTH_ACC": [(knn_preds_e == target_w)[month_len[i - 1]:month_len[i]].mean() for i in
                                range(1, month_len.shape[0])],
            "test_MONTH_ACC": [(knn_preds_e_test == target_w_test)[month_len[i - 1]:month_len[i]].mean() for i in
                               range(1, month_len.shape[0])],
            "train_CONFMAT": confusion_matrix(target_w, knn_preds_e).tolist(),
            "test_CONFMAT": confusion_matrix(target_w_test, knn_preds_e_test).tolist(),
            **{weekdays[i]: present_weekday(knn_preds_e == i, knn_preds_e_test == i, target_w == i, target_w_test == i) for i in range(8)},
            "working_day": present_weekday(knn_preds_e < 5, knn_preds_e_test < 5, target_w < 5, target_w_test < 5)
        }
    }
    return result


if __name__ == "__main__":
    Y_true, lengths, Y_test, lengths_test, target, target_test, target_w, target_w_test = provide_data()
    wandb_params, mstep_cofig, true_values = provide_logs()

    overall_result = collections.defaultdict(dict)
    overall_result["data"] = {
        "Y_true": Y_true.tolist(),
        "lengths": lengths.tolist(),
        "Y_test": Y_test.tolist(),
        "lengths_test": lengths_test.tolist(),
        "target": target.tolist(),
        "target_test": target_test.tolist(),
        "target_w": target_w.tolist(),
        "target_w_test": target_w_test.tolist()
    }

    for eta in lr:
        for n_, l_ in zip(n, l):
            for run in range(10):
                densehmm, states, states_test, loss, score, model_log, model_log2 = build_model(eta, n_, l_, Y_true, lengths, Y_test, lengths_test, wandb_params, mstep_cofig, true_values)
                overall_result[f"eta{eta}_n{n_}_l{l_}_run{run}"]["models"] = {"states_train": states.tolist(),
                                                                              "states_test": states_test.tolist(),
                                                                              "loss": loss,
                                                                              "score": score,
                                                                              "model_log": model_log2}
                overall_result[f"eta{eta}_n{n_}_l{l_}_run{run}"]["cluster_daily"] = [cluster_daily(model_log) for _ in range(10)]
                overall_result[f"eta{eta}_n{n_}_l{l_}_run{run}"]["cluster_weekly"] = [cluster_weekly(model_log, Y_true, Y_test, states, states_test, target, target_test) for _ in range(10)]
                overall_result[f"eta{eta}_n{n_}_l{l_}_run{run}"]["classify_working"] = {k_: [classify_working(k_, model_log, Y_true, Y_test, states, states_test, target, target_test) for _ in range(10)] for k_ in k}
                overall_result[f"eta{eta}_n{n_}_l{l_}_run{run}"]["classify_weekday"] = {k_: [classify_weekday(k_, model_log, Y_true, Y_test, states, states_test, target_w, target_w_test) for _ in range(10)] for k_ in k}

            with open(f"overall_result_eta{eta}_n{n_}_l{l_}_1d.json", "w") as f:
                json.dump(overall_result[f"eta{eta}_n{n_}_l{l_}_run{run}"], f, indent=4)

    with open(f"overall_result_1d.pkl", "wb") as f:
        pickle.dump(overall_result, f)
