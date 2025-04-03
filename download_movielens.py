from surprise import Dataset

# This will download and store the dataset locally in ~/.surprise
Dataset.load_builtin('ml-100k')
print("✅ MovieLens 100K downloaded!")
