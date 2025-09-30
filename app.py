from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import heapq

app = Flask(__name__)

# ================== Load Data ==================
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# ================== Collaborative Filtering ==================
user_movie_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
U, sigma, Vt = np.linalg.svd(user_movie_matrix, full_matrices=False)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
preds_df = pd.DataFrame(predicted_ratings, index=user_movie_matrix.index, columns=user_movie_matrix.columns)

def get_collaborative_based_recommendations(userId, movies_df, n=10):
    if userId not in preds_df.index:
        return [("User not found", 0.0)]
    sorted_user_preds = preds_df.loc[userId].sort_values(ascending=False)
    already_rated = ratings[ratings.userId == userId].movieId.values
    recommendations = []
    for movie_id, score in sorted_user_preds.items():
        if movie_id not in already_rated:
            movie_name = movies_df[movies_df.movieId == movie_id].title.values[0]
            recommendations.append((movie_name, score))
        if len(recommendations) >= n:
            break
    return recommendations

# ================== Content-Based Filtering ==================
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))
cosine_sim = cosine_similarity(tfidf_matrix)

def get_content_based_recommendations(title, movies_df, cosine_sim_matrix, n=10):
    title = title.strip().lower()
    matching_movies = movies_df[movies_df["title"].str.lower() == title]
    if matching_movies.empty:
        return ["Movie not found."]
    idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_n_movies = [movies_df.iloc[i[0]].title for i in sim_scores[1:n+1]]
    return top_n_movies

# ================== Improved DSA-Based Recommendations ==================
movie_ids = movies["movieId"].values
movie_index_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
genre_tfidf = TfidfVectorizer()
genre_matrix = genre_tfidf.fit_transform(movies["genres"].fillna(""))
genre_cosine_sim = cosine_similarity(genre_matrix)
avg_ratings = ratings.groupby("movieId")["rating"].mean().to_dict()

def get_dsa_based_recommendations(title, movies_df, ratings_df, n=10):
    title = title.strip().lower()
    matching_movies = movies_df[movies_df["title"].str.lower() == title]
    if matching_movies.empty:
        return ["Movie not found."]
    
    start_movie = matching_movies.iloc[0]["movieId"]
    start_idx = movie_index_map[start_movie]
    visited = set([start_movie])
    
    # Max-heap: (-score, movieId)
    heap = []
    
    for idx, m_id in enumerate(movie_ids):
        if m_id != start_movie:
            similarity = genre_cosine_sim[start_idx][idx]
            rating = avg_ratings.get(m_id, 0)
            score = similarity * rating
            heapq.heappush(heap, (-score, m_id))
    
    recommendations = []
    while heap and len(recommendations) < n:
        _, m_id = heapq.heappop(heap)
        if m_id not in visited:
            visited.add(m_id)
            movie_name = movies_df[movies_df.movieId == m_id].title.values[0]
            recommendations.append(movie_name)
    
    return recommendations if recommendations else ["No recommendations found."]

# ================== Routes ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare_recommendations():
    user_id = int(request.form["userId"])
    movie_title = request.form["movieTitle"]

    collaborative = get_collaborative_based_recommendations(user_id, movies, n=10)
    content = get_content_based_recommendations(movie_title, movies, cosine_sim, n=10)
    dsa = get_dsa_based_recommendations(movie_title, movies, ratings, n=10)

    return render_template("results.html", collaborative=collaborative, content=content, dsa=dsa)

if __name__ == "__main__":
    app.run(debug=True)
