# Movie Recommendation System

## Overview
This is a Flask-based Movie Recommendation System that uses **Collaborative Filtering** (SVD) and **Content-Based Filtering** (TF-IDF with cosine similarity) to recommend movies based on user preferences and movie genres.

## Features
- **Collaborative Filtering** using Singular Value Decomposition (SVD) from Surprise library.
- **Content-Based Filtering** using TF-IDF vectorization and cosine similarity.
- **Flask Web Interface** for user interaction.
- **Movie rating-based predictions** for personalized recommendations.
- **Genre-based similarity recommendations.**

## Technologies Used
- **Python** (Flask, Pandas, Scikit-learn, Surprise, Numpy)
- **Machine Learning** (Collaborative Filtering, TF-IDF, Cosine Similarity)
- **HTML, CSS, JavaScript** (for the web-based UI)

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- pip (Python package manager)

### Setup Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```
2. Install dependencies:
   ```bash
   pip install flask pandas scikit-learn surprise
   ```
3. Ensure you have `ratings.csv` and `movies.csv` datasets in the project directory.

4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Open your browser and go to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage
1. Open the web interface.
2. Enter your **User ID** to get personalized recommendations using **Collaborative Filtering**.
3. Enter a **Movie Title** to get similar movies using **Content-Based Filtering**.

## API Endpoints
### `GET /`
Renders the web-based UI.

### `POST /collaborative_recommend`
Generates recommendations based on user ratings using **SVD**.
- **Request Body:**
  ```json
  { "userId": 1 }
  ```
- **Response:**
  ```json
  [
    { "title": "Movie A", "predicted_rating": 4.5 },
    { "title": "Movie B", "predicted_rating": 4.3 }
  ]
  ```

### `POST /content_recommend`
Generates recommendations based on movie similarity using **TF-IDF & Cosine Similarity**.
- **Request Body:**
  ```json
  { "movieTitle": "Inception" }
  ```
- **Response:**
  ```json
  [
    "Interstellar",
    "The Matrix",
    "Shutter Island"
  ]
  ```

## Troubleshooting
- Ensure the dataset files (`ratings.csv` & `movies.csv`) are correctly placed.
- Check Python dependencies using `pip freeze`.
- If Flask fails to start, verify that port 5000 is not in use.

## License
This project is licensed under the MIT License.

## Author
**Your Name**  
GitHub: [Sudheerbmb](https://github.com/yourusername)
