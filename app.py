from flask import Flask, request, jsonify, render_template
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Load data (replace with actual path or load method)
ratings = pd.read_csv('ratings.csv')  # Path to your ratings dataset
movies = pd.read_csv('movies.csv')  # Path to your movies dataset

# Define rating scale
reader = Reader(rating_scale=(0.5, 5.0))

# Load dataset for collaborative filtering
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Train-test split
trainset, testset = train_test_split(data, test_size=0.2)

# Use Singular Value Decomposition (SVD) for collaborative filtering
svd_model = SVD()
svd_model.fit(trainset)

# Function to get top N recommendations using collaborative filtering
def get_collaborative_based_recommendations(model, userId, movies_df, n=10):
    movie_ids = movies_df['movieId'].unique()
    predictions = [model.predict(userId, movie_id) for movie_id in movie_ids]
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]
    recommended_movies = [(movies_df[movies_df.movieId == pred.iid].title.values[0], pred.est) for pred in top_n]
    return recommended_movies

# Convert genres into a TF-IDF matrix for content-based filtering
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["genres"].fillna(""))

# Compute cosine similarity scores for content-based filtering
cosine_sim = cosine_similarity(tfidf_matrix)

# Function to get content-based recommendations
def get_content_based_recommendations(title, movies_df, cosine_sim_matrix, n=10):
    title = title.strip().lower()  # Normalize input title (strip and lowercase)
    
    # Check if the title exists in the dataset
    matching_movies = movies_df[movies_df["title"].str.lower() == title]
    
    if matching_movies.empty:
        return ["Movie not found in the database. Please try another title."]
    
    idx = matching_movies.index[0]
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_n_movies = [movies_df.iloc[i[0]].title for i in sim_scores[1:n+1]]
    return top_n_movies

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/collaborative_recommend', methods=['POST'])
def collaborative_recommend():
    user_id = int(request.form['userId'])
    
    # Get top N recommendations using collaborative filtering
    recommendations = get_collaborative_based_recommendations(svd_model, userId=user_id, movies_df=movies, n=10)
    
    # Generate HTML content for collaborative recommendations page
    recommendations_html = "<h2>Top 10 Collaborative Filtering Recommendations</h2><ul>"
    for movie, rating in recommendations:
        recommendations_html += f"<li>{movie}: Predicted Rating {rating:.2f}</li>"
    recommendations_html += "</ul>"
    
    recommendations_html += '<a href="/">Back to Home</a>'

    return recommendations_html

@app.route('/content_recommend', methods=['POST'])
def content_recommend():
    movie_title = request.form['movieTitle']
    
    # Get top N content-based recommendations
    recommendations = get_content_based_recommendations(movie_title, movies, cosine_sim, n=10)
    
    # Generate HTML content for content-based recommendations page
    recommendations_html = "<h2>Top 10 Content-Based Recommendations</h2><ul>"
    for movie in recommendations:
        recommendations_html += f"<li>{movie}</li>"
    recommendations_html += "</ul>"
    
    recommendations_html += '<a href="/">Back to Home</a>'

    return recommendations_html

if __name__ == '__main__':
    app.run(debug=True)
