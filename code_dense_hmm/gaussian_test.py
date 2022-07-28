"""
Simple benchmark for GaussianDenseHMM

Chcemy sprawdzić:
- czy model radzi sobie z prostymi i trudnymi modelami
- czy model radzi sobie z dużymi i małymi porcjami danych
- jakie są łączne trajektorie embedingów
- jakie parametry są dobre do SGD w różnych metodach
- ile czasu zajmuje uczenie
- jak się skaluje czas dla różnych imlementacji
"""
import numpy as np
import itertools
from hmmlearn import hmm
from models_gaussian import StandardGaussianHMM, GaussianDenseHMM
import time
from tqdm import tqdm
from ssm.util import find_permutation
import pickle
import json
from pathlib import Path
import wandb

np.random.seed(2022)

# TODO: handle nans in dense, when the model collapses to less states
# TODO: logging to wandb
# TODO: recheck permuting embeddings
# TODO: decrease covergence tolerance

# TODO: log in wandb
# wandb.init(project="my-test-project",
#            entity="kabalce",
#            save_code=True,
#            group="test1",
#            job_type="test1.1",
#            name="testuje nazwę"  # nazwa wyświetlana na lewej stronie
#            )
#
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }
#
# for i in range(10):
#     wandb.log({"loss": i/10,  'kos': .454})  # różne klucze  na różnych wykresach, różne odpalenia na jednym wykresie

# TODO:  draw  pdfs
simple_model = {"mu": 10,
                "sigma": 1}

complicated_model = {"mu": 5,
                     "sigma": 2}

data_sizes = [(10, 12, 3), (10, 12, 5), (10,  100, 3), (10,  100, 5),  # with 2 iterations takes 9 minutes
              (100, 1000, 5), (100, 1000, 50)]  # (s, T, n)
ls = (2, 3, 5, 7, 10, 20)

mstep_cofigs = [{"em_lr": 0.0001, "em_epochs": 50},
                {"em_lr": 0.001, "em_epochs": 5},
                {"em_lr": 0.001, "em_epochs": 10},
                {"em_lr": 0.001, "em_epochs": 50},
                {"em_lr": 0.01, "em_epochs": 5},
                {"em_lr": 0.01, "em_epochs": 10},
                {"em_lr": 0.01, "em_epochs": 50},
                {"em_lr": 0.1, "em_epochs": 5},
                {"em_lr": 0.1, "em_epochs": 10}]

em_iters = [10, 25, 50]


def prepare_params(n, simple_model=True):
    pi = np.random.uniform(size=n)
    pi /= pi.sum()
    A = np.exp(np.random.uniform(0, 5, size=(n, n)))
    A /= A.sum(axis=1)[:, np.newaxis]

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


def dtv(a1, a2):
    return (abs(a1 - a2) / 2).sum() / a1.shape[0]


def apply_perm(a, perm):
    if a.shape[1] == 1:
        return a
    res = np.zeros_like(a)
    for i in range(a.shape[0]):
        res[perm[i], :] = a[i, :]
    return res

def permute_embeddings(embeddings, perm):
    return [
        apply_perm(embeddings[0], perm).tolist(),
        apply_perm(embeddings[1].transpose(), perm).transpose().tolist(),
        embeddings[2].tolist()
    ]


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
                 embeddings=permute_embeddings(embeddings, perm) if embeddings != None else None)]


def run_experiment(results_dir, simple_model=True):
    for dsize in tqdm(data_sizes, desc=f"SIMPLE={simple_model}"):
        s = dsize[0];  T = dsize[1]; n = dsize[2]
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

        for em_it in em_iters:
            # GaussianHMM - hmmlearn implementation
            hmml = hmm.GaussianHMM(n, n_iter=2, covariance_type='diag', init_params="", params="stmc")  # TODO!!  em_it
            hmml._init(Y_true)
            # Randomly chosen initial parameters
            A_init = hmml.transmat_
            pi_init = hmml.startprob_
            m_init = hmml.means_
            c_init = hmml._covars_

            start = time.perf_counter()
            hmml.fit(Y_true, lengths)
            time_tmp = time.perf_counter() - start

            preds = np.concatenate([hmml.predict(x[1]) for x in data])
            perm = find_permutation(preds, X_true)
            preds_perm = np.array([perm[i] for i in preds])
            result['hmmlearn_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
                                                        hmml,  time_tmp, perm, preds_perm)

            # GaussianHMM - custom implementation
            standardhmm = StandardGaussianHMM(n, em_iter=2, covariance_type='diag', init_params="", params="stmc")  # TODO!!  em_it
            standardhmm.transmat_ = A_init
            standardhmm.startprob_ = pi_init
            standardhmm.means_ = m_init
            standardhmm._covars_ = c_init

            start = time.perf_counter()
            standardhmm.fit(Y_true, lengths)
            time_tmp = time.perf_counter() - start

            preds = np.concatenate([standardhmm.predict(x[1]) for x in data])
            perm = find_permutation(preds, X_true)
            preds_perm = np.array([perm[i] for i in preds])
            result['hmmlearn_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
                                                        standardhmm, time_tmp, perm, preds_perm)

            # GaussianDenseHMM - custom implementation
            for mstep_cofig, l in itertools.product(mstep_cofigs, ls):
                # allow only embeddings of size smaller or equal the number of hidden states
                if l > n:
                    continue

                densehmm = GaussianDenseHMM(n, mstep_config={**mstep_cofig, "l_uz": l},
                                            covariance_type='diag', em_iter=2,  # TODO!! em_it
                                            init_params="", params="stmc")
                densehmm.transmat_ = A_init
                densehmm.startprob_ = pi_init
                densehmm.means_ = m_init
                densehmm._covars_ = c_init

                start = time.perf_counter()
                densehmm.fit(Y_true, lengths)
                time_tmp = time.perf_counter() - start

                preds = np.concatenate([densehmm.predict(x[1]) for x in data])
                perm = find_permutation(preds, X_true)
                preds_perm = np.array([perm[i] for i in preds])
                # TODO: zapisz embedingi
                result['hmmlearn_runs'] += provide_log_info(pi, A, mu, sigma, X_true,
                                                            densehmm, time_tmp, perm, preds_perm,
                                                            {**mstep_cofig, "l_uz": l},
                                                            embeddings=densehmm.get_representations())

        with open(f"{results_dir}/s={s}_T={T}_n={n}_simple={simple_model}.json", "w") as f:
            json.dump(result, f, indent=4)

    return None


if __name__ == "__main__":
    t = time.localtime()
    results_dir = f'gaussian_dense_hmm_benchmark/basic-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    run_experiment(results_dir, simple_model=True)
    run_experiment(results_dir, simple_model=False)
