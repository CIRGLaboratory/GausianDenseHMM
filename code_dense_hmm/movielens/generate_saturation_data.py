import pandas as pd
import numpy as np
import json
import time
import itertools
from funk_svd import SVD as FSVD
from pathlib import Path
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

# SETUP
GENRE1 = "Action"
GENRE2 = "Romance"

lr = 0.003308564402831481
reg = 0.04670255240414368
n_epochs = 83
n_factors = 81

t = time.localtime()
RESULT_DIR = f'../../data/benchmark_rs/saturation-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'

np.random.seed(2022)
no_cores = 10


def select_users(ratings):
    user_list = np.random.choice(ratings.u_id.unique(), 16, replace=False)
    with open(f"{RESULT_DIR}/users.json", "w") as f:
        json.dump(user_list.tolist(), f)
    return user_list


def provide_data():
    ratings = pd.read_csv('../../data/rating.csv').rename(columns={"userId": 'u_id', "movieId": "i_id"})
    movies = pd.read_csv('../../data/movie.csv').rename(columns={"movieId": "i_id"})
    genres = pd.DataFrame({k: {g: True for g in v} for k, v in
                           movies.set_index('i_id').genres.apply(lambda gs: gs.split("|")).to_dict().items()}).fillna(False).transpose()
    return ratings, movies, genres


def get_rating(df, u, i):
    tmp = df.loc[(df.u_id == u) & (df.i_id == i)]
    if tmp.empty:
        return np.nan
    else:
        return tmp.values[0]


def provide_ratings(train, test):
    fsvd = FSVD(lr=lr, reg=reg, n_epochs=n_epochs, n_factors=n_factors,
                early_stopping=True, shuffle=False, min_rating=1, max_rating=5)
    fsvd.fit(train)
    preds = fsvd.predict(test)
    return preds


def provide_all_available(scores, user):  # OK
    items = scores.i_id.drop_duplicates()
    all = pd.merge(pd.DataFrame([(u, i) for i, u in itertools.product(items, user)], columns=["u_id", "i_id"]),
                   scores.drop('timestamp', axis=1),
                   how='left', on=['u_id', 'i_id'])
    all_available = all.loc[all.rating.isna(), :].drop('rating', axis=1)
    return all_available


def sample_new_scores(available, genre, sample_size, g):  # OK
    available_all_genres = pd.merge(available,
                                    genre.loc[genre.sum(axis=1) < 3, :].reset_index().rename(columns={'index': 'i_id'}),
                                    how="left", on="i_id").fillna(False)
    available_genres = available_all_genres.loc[available_all_genres[g], ['u_id', 'i_id', 'pred']]
    new_scores = available_genres.groupby('u_id').apply(
        lambda df: pd.DataFrame(
            {'i_id': np.random.choice(df.sort_values('pred').i_id.values[-(sample_size * 2):], sample_size, replace=False),
             'rating': [5 for _ in range(sample_size)]})
    ).reset_index().drop('level_1', axis=1)
    return new_scores


def generate_saturation(args):
    rats, new_s, all_avail, step, iter = args
    new_scores_tmp = new_s.groupby('u_id').apply(lambda df: df.iloc[:step, :]).reset_index(drop=True)
    ratings_tmp = pd.concat([rats, new_scores_tmp], axis=0)
    all_available_tmp = all_avail.loc[~all_avail.u_id.isin(new_scores_tmp.u_id.values.tolist()) | ~all_avail.i_id.isin(new_scores_tmp.i_id.values.tolist()), :]

    preds = provide_ratings(ratings_tmp, all_available_tmp)

    res_tmp = pd.concat([all_available_tmp.reset_index(drop=True), pd.DataFrame({"pred": preds})], axis=1)
    res_tmp.to_parquet(f"{RESULT_DIR}/predictions_step_{iter * no_cores + step}.parquet")

    saturation = res_tmp.groupby("u_id").apply(
        lambda df: df.sort_values("pred")[-100:].drop(["u_id", "i_id", "pred"], axis=1).mean()).to_dict("index")

    return saturation


if __name__ == "__main__":
    Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)
    ratings, movies, genres = provide_data()
    users = select_users(ratings)
    genres_id = genres.reset_index().rename(columns={'index': 'i_id'})

    saturation_list = []

    for i in range(100):
        # Provide new scores
        gen = GENRE1 if i % 2 else GENRE2
        all_available = provide_all_available(ratings, users)
        all_available['pred'] = provide_ratings(ratings, all_available)
        new_scores = sample_new_scores(all_available, genres, no_cores, gen)
        all_available.drop('pred', axis=1, inplace=True)
        all_available = pd.merge(all_available, genres_id, how='left', on='i_id')

        with Pool(nodes=no_cores) as pool:
            saturation_list += pool.map(generate_saturation,
                                        [a for a in itertools.product([ratings], [new_scores], [all_available], range(no_cores), [i])])

        ratings = pd.concat([ratings, new_scores])

    with open(f"{RESULT_DIR}/all_saturation.json", "w") as f:
        json.dump(saturation_list, f, indent=4)
