import csv
import urllib.request
import io
import math
from collections import defaultdict

# ========== LOAD DATA ==========
def load_data():
    print("⏳ Loading MovieLens dataset...")
    ratings_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.data"
    movies_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"

    # Load ratings
    response = urllib.request.urlopen(ratings_url)
    data = csv.reader(io.TextIOWrapper(response), delimiter='\t')
    ratings = defaultdict(dict)
    for row in data:
        user_id = int(row[0])
        item_id = int(row[1])
        rating = float(row[2])
        ratings[user_id][item_id] = rating

    # Load movie titles
    movie_titles = {}
    response = urllib.request.urlopen(movies_url)
    movies_data = csv.reader(io.TextIOWrapper(response, encoding='latin1'), delimiter='|')
    for row in movies_data:
        movie_id = int(row[0])
        title = row[1]
        movie_titles[movie_id] = title

    print(f"✅ Loaded {sum(len(user_ratings) for user_ratings in ratings.values())} ratings")
    print(f"📊 Stats: {len(ratings)} users, {len(set(item for user in ratings.values() for item in user))} movies")

    return ratings, movie_titles

# ========== SIMILARITY ==========
def similarity(ratings, user1, user2):
    common_movies = set(ratings[user1]) & set(ratings[user2])
    if not common_movies:
        return 0

    sum1 = sum(ratings[user1][movie] for movie in common_movies)
    sum2 = sum(ratings[user2][movie] for movie in common_movies)

    sum1_sq = sum(pow(ratings[user1][movie], 2) for movie in common_movies)
    sum2_sq = sum(pow(ratings[user2][movie], 2) for movie in common_movies)

    p_sum = sum(ratings[user1][movie] * ratings[user2][movie] for movie in common_movies)

    n = len(common_movies)
    num = p_sum - (sum1 * sum2 / n)
    den = math.sqrt((sum1_sq - pow(sum1, 2) / n) * (sum2_sq - pow(sum2, 2) / n))

    return num / den if den != 0 else 0

# ========== RECOMMENDER ==========
def recommend(ratings, user_id, n=5):
    totals = defaultdict(float)
    similarity_sums = defaultdict(float)

    for other_user in ratings:
        if other_user == user_id:
            continue

        sim = similarity(ratings, user_id, other_user)
        if sim <= 0:
            continue

        for movie in ratings[other_user]:
            if movie not in ratings[user_id]:
                totals[movie] += ratings[other_user][movie] * sim
                similarity_sums[movie] += sim

    rankings = [(total / similarity_sums[movie], movie)
               for movie, total in totals.items()]
    rankings.sort(reverse=True)

    return rankings[:n]



from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# ========== SVD MODEL TRAINING ==========
def train_svd_model():
    # Reformat your ratings dict into Surprise-compatible format
    reader = Reader(line_format='user item rating timestamp', sep='\t')
    data = Dataset.load_builtin('ml-100k')  # Automatically downloads if needed
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train the model
    model = SVD()
    model.fit(trainset)
    predictions = model.test(testset)

    # Evaluate
    print("✅ SVD model trained")
    print("📉 RMSE:", accuracy.rmse(predictions))

    return model, trainset
