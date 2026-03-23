from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json

app = Flask(__name__)

# Load the dataset
movies_file = 'movie.csv'
ratings_file = 'ratings.csv'

movies_df = pd.read_csv(movies_file)
ratings_df = pd.read_csv(ratings_file, delimiter='\t')

# Clean data
movies_df.dropna(subset=['title', 'genres'], inplace=True)
ratings_df.dropna(subset=['userId', 'movieId', 'rating'], inplace=True)

# get release year from movies
movies_df['release_year'] = movies_df['title'].str.extract(r'\((\d{4})\)', expand=False).astype(float)
movies_df['movieId'] = movies_df['movieId'].astype(int)
ratings_df['movieId'] = ratings_df['movieId'].astype(int)

# Handle missing release years - fill with median
median_year = movies_df['release_year'].median()
movies_df['release_year'] = movies_df['release_year'].fillna(median_year)

# Get all unique genres
all_genres = sorted(movies_df['genres'].str.split('|').explode().dropna().unique())

# Create genre vectors for each movie using ML vectorization
def create_genre_vector(genre_string, all_genres):
    """Convert genre string to binary feature vector for ML similarity"""
    movie_genres = set(genre_string.split('|'))
    vector = np.array([1 if genre in movie_genres else 0 for genre in all_genres])
    return vector

# Vectorize all movies (one-time computation)
movies_df['genre_vector'] = movies_df['genres'].apply(lambda x: create_genre_vector(x, all_genres))

# Precompute movie ratings for KNN features
def compute_movie_ratings():
    """Compute average rating for each movie"""
    movie_ratings = ratings_df.groupby('movieId')['rating'].mean().to_dict()
    return movie_ratings

movie_avg_ratings = compute_movie_ratings()

# Function to recommend movies using ML: K-Nearest Neighbors (KNN)
def recommend_movies(selected_genres, preference, k=10):
    """
    ML Recommendation Algorithm: K-Nearest Neighbors (KNN)
    Finds K movies most similar to user preferences using Euclidean distance.
    Features: genres (binary), average rating (normalized), release year (normalized)
    No training required - pure algorithm computes distances at query time.
    """
    # Prepare feature matrix for all movies
    feature_vectors = []
    movie_list = []
    
    # Normalize rating and year ranges
    min_year = movies_df['release_year'].min()
    max_year = movies_df['release_year'].max()
    year_range = max_year - min_year if max_year > min_year else 1
    
    for idx, row in movies_df.iterrows():
        movie_id = row['movieId']
        
        # Genre features (binary vector)
        genre_vector = row['genre_vector'].astype(float)
        
        # Rating feature (normalized 0-1)
        avg_rating = movie_avg_ratings.get(movie_id, 2.5)
        normalized_rating = avg_rating / 5.0  # Scale to 0-1
        
        # Year feature (normalized 0-1)
        normalized_year = (row['release_year'] - min_year) / year_range if year_range > 0 else 0.5
        
        # Combine features: genres + rating + year
        combined_vector = np.concatenate([genre_vector, [normalized_rating, normalized_year]])
        feature_vectors.append(combined_vector)
        movie_list.append(movie_id)
    
    feature_matrix = np.array(feature_vectors)
    
    # Remove any NaN values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    # Create user preference vector with same dimensionality
    user_genre_vector = np.array([1 if genre in selected_genres else 0 for genre in all_genres]).astype(float)

    # Preference-aware targets and feature weighting.
    # This makes KNN distance itself sensitive to "recent" vs "high rating", not just final sorting.
    if preference == 'recent_release':
        normalized_user_rating = 0.70
        normalized_user_year = 1.0
        rating_weight = 0.8
        year_weight = 2.6
    elif preference == 'high_rating':
        normalized_user_rating = 1.0
        normalized_user_year = 0.6
        rating_weight = 2.0
        year_weight = 0.8
    else:
        normalized_user_rating = 0.75
        normalized_user_year = 0.7
        rating_weight = 1.0
        year_weight = 1.0

    # Combine user features
    user_vector = np.concatenate([user_genre_vector, [normalized_user_rating, normalized_user_year]])
    user_vector = user_vector.reshape(1, -1)
    
    # Ensure no NaN values
    user_vector = np.nan_to_num(user_vector, nan=0.0)
    
    # Apply preference weighting to feature space before KNN distance calculation
    feature_weights = np.ones(feature_matrix.shape[1], dtype=float)
    feature_weights[-2] = rating_weight
    feature_weights[-1] = year_weight
    weighted_feature_matrix = feature_matrix * feature_weights
    weighted_user_vector = user_vector * feature_weights

    # Apply KNN: find k nearest neighbors using weighted Euclidean distance
    knn = NearestNeighbors(n_neighbors=min(k, len(feature_matrix)), metric='euclidean', algorithm='auto')
    knn.fit(weighted_feature_matrix)
    
    distances, indices = knn.kneighbors(weighted_user_vector)
    
    # Get recommended movie IDs
    knn_movie_ids = [movie_list[i] for i in indices[0]]
    knn_distances = distances[0]
    
    # Get movie details and ratings
    recommendations_data = []
    for movie_id, distance in zip(knn_movie_ids, knn_distances):
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        recommendations_data.append({
            'movieId': movie_id,
            'title': movie_info['title'],
            'release_year': movie_info['release_year'],
            'distance': distance
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    
    # Get average ratings for recommended movies
    movie_ratings_merged = pd.merge(
        ratings_df[ratings_df['movieId'].isin(recommendations_df['movieId'])],
        recommendations_df,
        on='movieId',
        how='outer'  # Use outer join to keep all recommended movies
    )
    
    avg_ratings = movie_ratings_merged.groupby('movieId').agg({
        'rating': 'mean',
        'title': 'first',
        'release_year': 'first',
        'distance': 'first'
    }).reset_index()
    
    avg_ratings = avg_ratings.rename(columns={'rating': 'avg_rating'})
    # Fill NaN ratings with 3.0 for movies without ratings
    avg_ratings['avg_rating'] = avg_ratings['avg_rating'].fillna(3.0).round()
    
    # Sort by distance (closer = better) which is the KNN ranking
    # Secondary sort by user preference
    if preference == 'high_rating':
        avg_ratings = avg_ratings.sort_values(by=['distance', 'avg_rating'], ascending=[True, False])
    else:
        avg_ratings = avg_ratings.sort_values(by=['distance', 'release_year'], ascending=[True, False])
    
    # Return top 10 with KNN distance scores
    return avg_ratings[['title', 'avg_rating', 'release_year', 'distance']].head(10)

# Function to get KNN recommendations with visualization data
def recommend_movies_with_viz(selected_genres, preference, k=10):
    """
    KNN with visualization data for frontend demo.
    Returns recommendations + visualization metrics.
    """
    # Prepare feature matrix for all movies
    feature_vectors = []
    movie_list = []
    movie_titles = []
    
    # Normalize rating and year ranges
    min_year = movies_df['release_year'].min()
    max_year = movies_df['release_year'].max()
    year_range = max_year - min_year if max_year > min_year else 1
    
    for idx, row in movies_df.iterrows():
        movie_id = row['movieId']
        genre_vector = row['genre_vector'].astype(float)
        avg_rating = movie_avg_ratings.get(movie_id, 2.5)
        normalized_rating = avg_rating / 5.0
        normalized_year = (row['release_year'] - min_year) / year_range if year_range > 0 else 0.5
        
        combined_vector = np.concatenate([genre_vector, [normalized_rating, normalized_year]])
        feature_vectors.append(combined_vector)
        movie_list.append(movie_id)
        movie_titles.append(row['title'])
    
    feature_matrix = np.array(feature_vectors)
    
    # Remove any NaN values that might exist
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
    
    # Create user preference vector
    user_genre_vector = np.array([1 if genre in selected_genres else 0 for genre in all_genres]).astype(float)

    # Preference-aware targets and feature weighting
    if preference == 'recent_release':
        normalized_user_rating = 0.70
        normalized_user_year = 1.0
        rating_weight = 0.8
        year_weight = 2.6
    elif preference == 'high_rating':
        normalized_user_rating = 1.0
        normalized_user_year = 0.6
        rating_weight = 2.0
        year_weight = 0.8
    else:
        normalized_user_rating = 0.75
        normalized_user_year = 0.7
        rating_weight = 1.0
        year_weight = 1.0
    
    user_vector = np.concatenate([user_genre_vector, [normalized_user_rating, normalized_user_year]])
    user_vector = user_vector.reshape(1, -1)
    
    # Ensure no NaN values in user vector
    user_vector = np.nan_to_num(user_vector, nan=0.0)
    
    # Apply preference weighting before KNN distance calculation
    feature_weights = np.ones(feature_matrix.shape[1], dtype=float)
    feature_weights[-2] = rating_weight
    feature_weights[-1] = year_weight
    weighted_feature_matrix = feature_matrix * feature_weights
    weighted_user_vector = user_vector * feature_weights

    # Apply KNN
    knn = NearestNeighbors(n_neighbors=min(k, len(feature_matrix)), metric='euclidean', algorithm='auto')
    knn.fit(weighted_feature_matrix)
    distances, indices = knn.kneighbors(weighted_user_vector)
    
    # Get all distances (not just top k) for visualization
    knn_all = NearestNeighbors(n_neighbors=len(feature_matrix), metric='euclidean', algorithm='auto')
    knn_all.fit(weighted_feature_matrix)
    all_distances, all_indices = knn_all.kneighbors(weighted_user_vector)
    
    # Prepare visualization data
    all_distances_flat = all_distances[0]
    all_movies_sorted = [(movie_list[i], movie_titles[i], all_distances_flat[idx]) 
                         for idx, i in enumerate(all_indices[0])]
    
    # Get recommended movie IDs
    knn_movie_ids = [movie_list[i] for i in indices[0]]
    knn_distances = distances[0]
    
    # Get movie details
    recommendations_data = []
    for movie_id, distance in zip(knn_movie_ids, knn_distances):
        movie_info = movies_df[movies_df['movieId'] == movie_id].iloc[0]
        recommendations_data.append({
            'movieId': movie_id,
            'title': movie_info['title'],
            'release_year': movie_info['release_year'],
            'distance': float(distance)
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    
    # Get average ratings
    movie_ratings_merged = pd.merge(
        ratings_df[ratings_df['movieId'].isin(recommendations_df['movieId'])],
        recommendations_df,
        on='movieId',
        how='outer'
    )
    
    avg_ratings = movie_ratings_merged.groupby('movieId').agg({
        'rating': 'mean',
        'title': 'first',
        'release_year': 'first',
        'distance': 'first'
    }).reset_index()
    
    avg_ratings = avg_ratings.rename(columns={'rating': 'avg_rating'})
    avg_ratings['avg_rating'] = avg_ratings['avg_rating'].fillna(3.0).round()
    
    if preference == 'high_rating':
        avg_ratings = avg_ratings.sort_values(by=['distance', 'avg_rating'], ascending=[True, False])
    else:
        avg_ratings = avg_ratings.sort_values(by=['distance', 'release_year'], ascending=[True, False])
    
    recommendations = avg_ratings[['title', 'avg_rating', 'release_year', 'distance']].head(10)
    
    # Prepare visualization data
    viz_data = {
        'all_distances': [(title, float(d)) for movie_id, title, d in all_movies_sorted],
        'top_10_indices': list(range(10)),  # First 10 are top matches
        'selected_genres': selected_genres,
        'total_movies': len(feature_matrix),
        'feature_count': len(user_vector[0]),
        'user_vector_sample': [float(x) for x in user_vector[0][:min(5, len(user_vector[0]))]],
    }
    
    return recommendations, viz_data

# Flask Routes
@app.route('/')
def index():
    # Get unique genres from the movies dataset
    genres = all_genres
    return render_template('index.html', genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_genres = request.form.getlist('genres')
    preference = request.form.get('preference')
    preference_label = 'High Rating' if preference == 'high_rating' else 'Recent Release'
    
    if not selected_genres:
        return render_template('error.html', error="Please select at least one genre.")
    
    recommended_movies = recommend_movies(selected_genres, preference)
    
    if not recommended_movies.empty:
        # Get visualization data as well
        _, viz_data = recommend_movies_with_viz(selected_genres, preference)
        movies_list = recommended_movies.to_dict(orient='records')
        return render_template(
            'recommendations.html',
            movies=movies_list,
            viz_data=viz_data,
            preference=preference,
            preference_label=preference_label
        )
    else:
        return render_template('error.html', error="No recommendations found for the selected genres.")

@app.route('/recommend_viz', methods=['POST'])
def recommend_viz():
    selected_genres = request.form.getlist('genres')
    preference = request.form.get('preference')
    preference_label = 'High Rating' if preference == 'high_rating' else 'Recent Release'
    
    if not selected_genres:
        return render_template('error.html', error="Please select at least one genre.")
    
    recommended_movies, viz_data = recommend_movies_with_viz(selected_genres, preference)
    
    if not recommended_movies.empty:
        movies_list = recommended_movies.to_dict(orient='records')
        return render_template(
            'recommendations_viz.html',
            movies=movies_list,
            viz_data=viz_data,
            preference=preference,
            preference_label=preference_label
        )
    else:
        return render_template('error.html', error="No recommendations found for the selected genres.")

@app.route('/how_to_use')
def how_it_works():
    return render_template('how_it_works.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
