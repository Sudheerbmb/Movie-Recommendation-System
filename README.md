* **Collaborative Filtering (SVD):**
    * Unearth hidden gems based on user ratings. We employ Singular Value Decomposition (SVD) to predict your cinematic soulmate films. 💖
    * Input your `userId` and witness the magic!
* **Content-Based Filtering (TF-IDF):**
    * Discover movies that resonate with your favorite genres. We leverage TF-IDF and cosine similarity to find films that share the essence of your chosen title. 🎭
    * Input a `movieTitle` and let the recommendations roll!

**How to Conjure the Recommendations:**

1.  **Clone the Repository:** `git clone [repository_url]`
2.  **Install the Cast:** `pip install Flask scikit-learn pandas surprise`
3.  **Prepare the Stage:** Place your `ratings.csv` and `movies.csv` files in the same directory as `app.py`.
4.  **Begin the Show:** `python app.py`
5.  **Enter the Theatre:** Open your browser and navigate to `http://127.0.0.1:5000/`.
6.  **Choose Your Adventure:**
    * For collaborative recommendations, enter a `userId`.
    * For content-based recommendations, enter a `movieTitle`.
7.  **Let the Reels Roll!** Receive your personalized movie recommendations. 🎥

**Technical Breakdown (Behind the Curtain):**

* **Collaborative Filtering:**
    * Utilizes the `surprise` library for SVD, a powerful matrix factorization technique.
    * Trains a model on user-movie ratings to predict unseen ratings.
* **Content-Based Filtering:**
    * Employs `sklearn`'s `TfidfVectorizer` to convert movie genres into a numerical representation.
    * Calculates cosine similarity to find movies with similar genre profiles.

**Why Choose ReelRec?**

* **Dual Recommendation Power:** Enjoy the best of both worlds with collaborative and content-based filtering. 🌐
* **User-Friendly Interface:** Simple and intuitive web interface powered by Flask. 💻
* **Efficient and Accurate:** Leveraging proven algorithms for reliable recommendations. ✅

**Contributing (Join the Crew):**

Feel free to enhance ReelRec with new features, optimizations, or improvements. Fork the repository and submit your pull requests. Let's make movie discovery magical! 🤝

**Disclaimer (The Credits):**

This application relies on the `ratings.csv` and `movies.csv` datasets. Ensure data integrity for optimal performance. ⚠️
