from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import heapq

app = Flask(__name__)

# ================== Load Data ==================
ratings = pd.read_csv("ratings.csv")  # userId, movieId, rating
movies = pd.read_csv("movies.csv")    # movieId, title, genres

# ================== Collaborative Filtering (Matrix Factorization) ==================
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
        return ["Movie not found in the database. Please try another title."]
    idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_n_movies = [movies_df.iloc[i[0]].title for i in sim_scores[1:n+1]]
    return top_n_movies

# ================== DSA-Based Recommendations (Graph + BFS + Heap) ==================
graph = defaultdict(list)
movie_genres = {}

for _, row in movies.iterrows():
    movie_id = row["movieId"]
    genres = row["genres"].split("|") if pd.notna(row["genres"]) else []
    movie_genres[movie_id] = genres
    for genre in genres:
        graph[genre].append(movie_id)

def get_dsa_based_recommendations(title, movies_df, ratings_df, n=10):
    title = title.strip().lower()
    matching_movies = movies_df[movies_df["title"].str.lower() == title]
    if matching_movies.empty:
        return ["Movie not found in the database. Please try another title."]
    movie_id = matching_movies.iloc[0]["movieId"]

    if movie_id not in movie_genres:
        return ["No genre information available for this movie."]

    genres = movie_genres[movie_id]
    visited = set()
    candidate_movies = []

    for genre in genres:
        queue = deque(graph[genre])
        while queue:
            neighbor = queue.popleft()
            if neighbor != movie_id and neighbor not in visited:
                visited.add(neighbor)
                avg_rating = ratings_df[ratings_df.movieId == neighbor]["rating"].mean()
                if not np.isnan(avg_rating):
                    heapq.heappush(candidate_movies, (-avg_rating, neighbor))

    recommendations = []
    while candidate_movies and len(recommendations) < n:
        _, m_id = heapq.heappop(candidate_movies)
        movie_name = movies_df[movies_df.movieId == m_id].title.values[0]
        recommendations.append(movie_name)

    return recommendations if recommendations else ["No recommendations found."]

# ================== Routes ==================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/collaborative_recommend", methods=["POST"])
def collaborative_recommend():
    user_id = int(request.form["userId"])
    recommendations = get_collaborative_based_recommendations(userId=user_id, movies_df=movies, n=10)
    recommendations_html = "<h2>Top 10 Collaborative Filtering Recommendations</h2><ul>"
    for movie, rating in recommendations:
        recommendations_html += f"<li>{movie}: Predicted Rating {rating:.2f}</li>"
    recommendations_html += "</ul><a href='/'>Back to Home</a>"
    return recommendations_html

@app.route("/content_recommend", methods=["POST"])
def content_recommend():
    movie_title = request.form["movieTitle"]
    recommendations = get_content_based_recommendations(movie_title, movies, cosine_sim, n=10)
    recommendations_html = "<h2>Top 10 Content-Based Recommendations</h2><ul>"
    for movie in recommendations:
        recommendations_html += f"<li>{movie}</li>"
    recommendations_html += "</ul><a href='/'>Back to Home</a>"
    return recommendations_html

@app.route("/dsa_recommend", methods=["POST"])
def dsa_recommend():
    movie_title = request.form["movieTitleDSA"]
    recommendations = get_dsa_based_recommendations(movie_title, movies, ratings, n=10)
    recommendations_html = "<h2>Top 10 DSA-Based Recommendations</h2><ul>"
    for movie in recommendations:
        recommendations_html += f"<li>{movie}</li>"
    recommendations_html += "</ul><a href='/'>Back to Home</a>"
    return recommendations_html

if __name__ == "__main__":
    app.run(debug=True)
