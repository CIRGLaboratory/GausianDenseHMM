import pandas as pd
import numpy as np
import optuna
import joblib
import time
import json

from sklearn.metrics import mean_squared_error
from funk_svd import SVD as FSVD
from pathlib import Path


t = time.localtime()
RESULT_DIR = f'../../data/benchmark_rs/optimize-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'


def train_eval_svd(trial, ratings_train, ratings_test):
    n_epochs = trial.suggest_int('n_epochs', 10, 500)
    lr = trial.suggest_loguniform("lr", 1e-3, .75)
    reg = trial.suggest_loguniform("lr", 1e-4, .75)
    n_factors = trial.suggest_int('n_factors', 3, 100)

    fsvd = FSVD(lr=lr, reg=reg, n_epochs=n_epochs, n_factors=n_factors,
                early_stopping=True, shuffle=False, min_rating=1, max_rating=5)
    fsvd.fit(X=ratings_train)

    preds = fsvd.predict(ratings_test)

    if np.isnan(preds).sum() > 0:
        return 1000

    rmse = np.sqrt(mean_squared_error(ratings_test['rating'], preds))

    with open(f"{RESULT_DIR}/params_all.json", "a") as f:
        json.dump(dict(lr=lr, reg=reg, n_epochs=n_epochs, n_factors=n_factors, rmse=rmse), f, indent=4)

    return rmse


if __name__ == "__main__":
    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)
    print(Path(RESULT_DIR).absolute())

    ratings = pd.read_csv("../../data/rating.csv")
    ratings = ratings.rename(columns={"userId": 'u_id', "movieId": "i_id"})

    test_index = ratings.sort_values('timestamp').groupby(['u_id']).tail(3).index
    ratings_train = ratings.drop(test_index)
    ratings_test = ratings.loc[test_index, :]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: train_eval_svd(trial, ratings_train, ratings_test), n_trials=256)

    print(f"Best params: {study.best_params}")

    with open(f"{RESULT_DIR}/study.z", "wb") as f:
        joblib.dump(study, f)
    with open(f"{RESULT_DIR}/params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
