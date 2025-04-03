from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import pandas as pd
import urllib.request
import io
import csv

# ===== LOAD MOVIE TITLES =====
def load_movie_titles():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    response = urllib.request.urlopen(url)
    data = csv.reader(io.TextIOWrapper(response, encoding='latin1'), delimiter='|')

    movie_titles = {}
    for row in data:
        movie_id = row[0]
        title = row[1]
        movie_titles[movie_id] = title
    return movie_titles

# ===== TRAIN SVD MODEL =====
def train_model():
    data = Dataset.load_builtin('ml-100k')  # loads u.data automatically
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)

    rmse = accuracy.rmse(predictions)
    return model, predictions, trainset

# ===== TOP-N RECOMMENDER =====
def get_top_n(model, trainset, user_id, n=5):
    # Get all movie IDs
    all_items = trainset.all_items()
    raw_ids = [trainset.to_raw_iid(iid) for iid in all_items]

    # Predict ratings for movies user hasn't rated yet
    user_inner_id = trainset.to_inner_uid(str(user_id))
    user_rated = set([trainset.to_raw_iid(iid) for (iid, _) in trainset.ur[user_inner_id]])
    to_predict = [iid for iid in raw_ids if iid not in user_rated][:50]

    predictions = [model.predict(str(user_id), iid) for iid in to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions[:n]
