from app.svd_model import train_model, get_top_n, load_movie_titles

print("📥 Loading and training model...")
model, _, trainset = train_model()
print("✅ Model trained!")

print("🎬 Loading movie titles...")
movie_titles = load_movie_titles()
print("✅ Movie titles loaded!")

user_id = 196
print(f"🔍 Getting top-N recommendations for User {user_id}...")
top_n = get_top_n(model, trainset, user_id)
print("✅ Recommendations ready!\n")

print(f"🎬 Top 5 SVD-Based Recommendations for User {user_id}:")
for pred in top_n:
    title = movie_titles.get(pred.iid, f"Movie {pred.iid}")
    print(f"- {title} (Predicted Rating: {pred.est:.2f})")
