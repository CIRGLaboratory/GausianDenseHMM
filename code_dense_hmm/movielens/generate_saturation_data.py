import pandas as pd
import numpy as np
import json
import time
from funk_svd import SVD as FSVD

GENRE1 = "Action"
GENRE2 = "Romance"

lr = 0.00010676587674837098
reg = 0.00010676587674837098
n_epochs = 779
n_factors = 10

t = time.localtime()
RESULT_DIR = f'../../data/benchmark_rs/saturation-{t.tm_year}-{t.tm_mon}-{t.tm_mday}'


def get_rating(df, u, i):
    tmp = df.loc[(df.u_id == u) & (df.i_id == i)]
    if tmp.empty:
        return np.nan
    else:
        return tmp.values[0]


if __name__ == "__main__":
    ratings = pd.read_csv('../../data/rating.csv').rename(columns={"userId": 'u_id',  "movieId": "i_id"})
    movies = pd.read_csv('../../data/movie.csv').rename(columns={"movieId": "i_id"})
    genres = pd.DataFrame({k: {g: True for g in v} for k, v in movies.set_index('i_id').genres.apply(lambda gs: gs.split("|")).to_dict().items()}).fillna(False).transpose()

    np.random.seed(2022)

    users = np.random.choice(ratings.userId.unique(), 16, replace=False)
    all_movies = genres.index.values

    movies_available = pd.DataFrame([
        {"u_id": u,
         "i_id": i,
         "rating_real": get_rating(ratings, u, i)}
        for u in users for i in all_movies])
    movies_available = pd.concat([movies_available, genres.loc[movies_available.i_id].reset_index(drop=True)], axis=1)

    saturation_list = []

    for i in range(1000):
        fsvd = FSVD(lr=lr, reg=reg, n_epochs=n_epochs, n_factors=n_factors,
                    early_stopping=True, shuffle=False, min_rating=1, max_rating=5)
        fsvd.fit(ratings)
        preds = fsvd.predict(movies_available)

        # calculate the saturation
        res_tmp = pd.concat([movies_available, pd.DataFrame({"pred": preds})], axis=1)
        with open(f"{RESULT_DIR}/predictions_step_{i}.json", "w") as f:
            json.dump(res_tmp, f)

        saturation = res_tmp.groupby("u_id").apply(lambda df:  df.sort_values("pred")[-100:].drop(["u_id", "i_id",  "pred"],  axis=1).mean()).to_dict("index")

        # append saturation to file
        saturation_list.append(saturation)

        # Add  selected movies to ratings list with 4s and 5s
        genre_tmp = GENRE1 if (i % 50) < 25 else GENRE2

        selected = movies_available.loc[movies_available.i_id.isin(genres.index[genres[genre_tmp]]), :].groupby("u_id").apply(lambda df: np.random.choice(df.index))
        new_scores = pd.concat([movies_available.loc[selected, ["u_id", "i_id"]].reset_index(drop=True),
                                pd.DataFrame({"rating": np.random.choice([4, 5], selected.shape[0], p=[.15, .85])})],
                               axis=1)
        movies_available.loc[selected, 'rating'] = new_scores.rating.values
        ratings = pd.concat([ratings, new_scores], axis=0)

    with open(f"{RESULT_DIR}/all_saturation.json", "w") as f:
        json.dump(saturation_list,  f)
