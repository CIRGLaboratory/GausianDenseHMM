import pandas as pd
import numpy as np
import json
import time
from funk_svd import SVD as FSVD
from pathlib import Path
from tqdm import tqdm
from pathos.multiprocessing import ProcessingPool as Pool

GENRE1 = "Action"
GENRE2 = "Romance"

lr = 0.001942951202698156
reg = 0.051518838687760575
n_epochs = 404
# n_epochs = 2
n_factors = 90

t = time.localtime()
RESULT_DIR = f'../../data/benchmark_rs/test-saturation-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'


def get_rating(df, u, i):
    tmp = df.loc[(df.u_id == u) & (df.i_id == i)]
    if tmp.empty:
        return np.nan
    else:
        return tmp.values[0]


def generate_saturation(ratings, movies_available,  step, new_scores_tmp):
    ratings_tmp = pd.concat([ratings, new_scores_tmp], axis=0)
    movies_available_tmp = movies_available.copy()
    movies_available_tmp = movies_available_tmp.set_index(['u_id', 'i_id']).combine_first(new_scores_tmp.set_index(['u_id', 'i_id'])).reset_index()

    fsvd = FSVD(lr=lr, reg=reg, n_epochs=n_epochs, n_factors=n_factors,
                early_stopping=True, shuffle=False, min_rating=1, max_rating=5)
    fsvd.fit(ratings_tmp)
    preds = fsvd.predict(movies_available_tmp)

    # calculate the saturation
    res_tmp = pd.concat([movies_available_tmp, pd.DataFrame({"pred": preds})], axis=1)
    res_tmp.to_parquet(f"{RESULT_DIR}/predictions_step_{step}.parquet")

    print(res_tmp)

    saturation = res_tmp.groupby("u_id").apply(
        lambda df: df.sort_values("pred")[-100:].drop(["u_id", "i_id", "pred"], axis=1).mean()).to_dict("index")

    return saturation


# if __name__ == "__main__":
print("START")

Path(RESULT_DIR).mkdir(exist_ok=True, parents=True)
ratings = pd.read_csv('../../data/rating.csv').rename(columns={"userId": 'u_id',  "movieId": "i_id"})
movies = pd.read_csv('../../data/movie.csv').rename(columns={"movieId": "i_id"})
genres = pd.DataFrame({k: {g: True for g in v} for k, v in movies.set_index('i_id').genres.apply(lambda gs: gs.split("|")).to_dict().items()}).fillna(False).transpose()

print("Read data - DONE")

np.random.seed(2022)

users = np.random.choice(ratings.u_id.unique(), 16, replace=False)
with open(f"{RESULT_DIR}/users.json", "w") as f:
    json.dump(users.tolist(), f)

all_movies = genres.index.values

movies_tmp = ratings.loc[ratings.u_id.isin(users)].pivot('u_id', 'i_id', 'rating')
movies_available = pd.melt(movies_tmp.reset_index(),
                           id_vars='u_id',
                           value_vars=movies_tmp.columns,
                           var_name='i_id', value_name='rating_real')

movies_available = pd.concat([movies_available, genres.loc[movies_available.i_id].reset_index(drop=True)], axis=1)

print("Prepare variables - DONE")

saturation_list = []

pool = Pool(25)

for i in tqdm(range(40)):
# for i in tqdm(range(2)):
    genre_tmp = GENRE1 if i % 2 else GENRE2
    #  provide new scores for 25 iterations
    fsvd = FSVD(lr=lr, reg=reg, n_epochs=n_epochs, n_factors=n_factors,
                early_stopping=True, shuffle=False, min_rating=1, max_rating=5)
    fsvd.fit(ratings)
    preds = fsvd.predict(movies_available)

    # calculate the saturation
    movies_available = pd.concat([movies_available, pd.DataFrame({"pred": preds})], axis=1)

    selected = movies_available.loc[movies_available.i_id.isin(genres.index[genres[genre_tmp]]), :].groupby("u_id").apply(lambda df: pd.Series(np.random.choice(df.sort_values("pred")[-100:].index, 25, False))).melt().value
    new_scores = pd.concat([movies_available.loc[selected, ["u_id", "i_id"]].reset_index(drop=True),
                            pd.DataFrame({"rating": np.random.choice([4, 5], selected.shape[0], p=[.15, .85])})],
                           axis=1)

    def task(j, new_scores=new_scores, i=i):
        new_scores_tmp = new_scores.groupby("u_id").apply(lambda df: df[:(j + 1)])
        # print(new_scores_tmp.shape)
        saturation = generate_saturation(ratings, movies_available, i * 25 + j, new_scores_tmp)
        return saturation


    movies_available.drop("pred", axis=1, inplace=True)

    saturation_list += pool.map(task, list(range(25)))

    movies_available.loc[selected, 'rating'] = new_scores.rating.values
    ratings = pd.concat([ratings, new_scores], axis=0)

with open(f"{RESULT_DIR}/all_saturation.json", "w") as f:
    json.dump(saturation_list, f, indent=4)
