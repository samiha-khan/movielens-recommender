from app.model import load_data, recommend

ratings, movie_titles = load_data()

sample_user = 196
print(f"\n🎬 Top 5 Recommendations for User {sample_user}:")
for score, movie in recommend(ratings, sample_user):
    print(f"- {movie_titles.get(movie, f'Movie {movie}')} (score: {score:.2f})")
