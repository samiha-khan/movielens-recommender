# 🎬 MovieLens Recommender System

This project is a real-time movie recommendation engine built using the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/). It uses collaborative filtering via **Singular Value Decomposition (SVD)** and is deployed live using **FastAPI + Render** with a minimal frontend interface.

👉 **Live Demo**: [https://movielens-api.onrender.com](https://movielens-api.onrender.com)

---

## 🧠 Features

- ✅ Collaborative filtering using `scikit-surprise`'s SVD
- ✅ Recommends top-N movies for a given user
- ✅ Includes a simple web interface (HTML + JS)
- ✅ Live API endpoint at `/recommend/{user_id}?n=5`
- ✅ Auto-downloads dataset for cloud deployment

---

## 🚀 Tech Stack

- FastAPI (backend server)
- Jinja2 (templating engine)
- scikit-surprise (SVD recommender)
- Render.com (hosting)
- HTML + JS (frontend interface)

---

## 📦 Local Setup

```bash
git clone https://github.com/samiha-khan/movielens-recommender.git
cd movielens-recommender
pip install -r requirements.txt
uvicorn app.api:app --reload
