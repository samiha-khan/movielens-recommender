from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.svd_model import train_model, get_top_n, load_movie_titles

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")

# Train model + load data
print("⚙️ Training model, please wait...")
model, _, trainset = train_model()
movie_titles = load_movie_titles()
print("✅ Model loaded!")

# Route for UI
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API route
@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 5):
    top_preds = get_top_n(model, trainset, user_id, n)
    results = [
        {
            "movie_id": pred.iid,
            "title": movie_titles.get(pred.iid, f"Movie {pred.iid}"),
            "predicted_rating": round(pred.est, 2)
        }
        for pred in top_preds
    ]
    return {"user_id": user_id, "recommendations": results}
