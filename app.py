from flask import Flask, render_template, request, jsonify
from functools import lru_cache
import pandas as pd
import numpy as np

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

PREFERENCE_SETTINGS = {
    'recent_release': {
        'normalized_user_rating': 0.70,
        'normalized_user_year': 1.0,
        'rating_weight': 0.8,
        'year_weight': 2.6,
    },
    'high_rating': {
        'normalized_user_rating': 1.0,
        'normalized_user_year': 0.6,
        'rating_weight': 2.0,
        'year_weight': 0.8,
    },
    'default': {
        'normalized_user_rating': 0.75,
        'normalized_user_year': 0.7,
        'rating_weight': 1.0,
        'year_weight': 1.0,
    },
}


def _build_movie_feature_cache():
    """Build the movie feature matrix and lookup tables once at startup."""
    min_year = movies_df['release_year'].min()
    max_year = movies_df['release_year'].max()
    year_range = max_year - min_year if max_year > min_year else 1

    feature_vectors = []
    movie_ids_local = []
    movie_metadata_by_id_local = {}

    for row in movies_df.itertuples(index=False):
        movie_id = int(row.movieId)
        genre_vector = np.asarray(row.genre_vector, dtype=float)
        avg_rating = movie_avg_ratings.get(movie_id, 2.5)
        normalized_rating = avg_rating / 5.0
        normalized_year = (float(row.release_year) - min_year) / year_range if year_range > 0 else 0.5

        combined_vector = np.concatenate([genre_vector, [normalized_rating, normalized_year]])
        feature_vectors.append(combined_vector)
        movie_ids_local.append(movie_id)
        movie_metadata_by_id_local[movie_id] = {
            'title': row.title,
            'release_year': float(row.release_year),
        }

    feature_matrix = np.nan_to_num(np.array(feature_vectors, dtype=float), nan=0.0)

    feature_weights_by_preference_local = {}
    weighted_feature_matrices_by_preference_local = {}
    for preference_name, settings in PREFERENCE_SETTINGS.items():
        weights = np.ones(feature_matrix.shape[1], dtype=float)
        weights[-2] = settings['rating_weight']
        weights[-1] = settings['year_weight']
        feature_weights_by_preference_local[preference_name] = weights
        weighted_feature_matrices_by_preference_local[preference_name] = feature_matrix * weights

    return (
        feature_matrix,
        movie_ids_local,
        movie_metadata_by_id_local,
        feature_weights_by_preference_local,
        weighted_feature_matrices_by_preference_local,
    )


base_feature_matrix, movie_ids, movie_metadata_by_id, feature_weights_by_preference, weighted_feature_matrices_by_preference = _build_movie_feature_cache()


def _get_preference_settings(preference):
    return PREFERENCE_SETTINGS.get(preference, PREFERENCE_SETTINGS['default'])


def _normalize_selected_genres(selected_genres):
    return tuple(sorted(set(selected_genres)))


def _build_user_vector(selected_genres, preference):
    settings = _get_preference_settings(preference)
    user_genre_vector = np.array([1 if genre in selected_genres else 0 for genre in all_genres], dtype=float)
    user_vector = np.concatenate([
        user_genre_vector,
        [settings['normalized_user_rating'], settings['normalized_user_year']],
    ]).reshape(1, -1)
    user_vector = np.nan_to_num(user_vector, nan=0.0)
    return user_vector, settings


def _score_movies(selected_genres, preference):
    user_vector, settings = _build_user_vector(selected_genres, preference)
    weights = feature_weights_by_preference.get(preference, feature_weights_by_preference['default'])
    weighted_user_vector = user_vector * weights
    weighted_feature_matrix = weighted_feature_matrices_by_preference.get(
        preference,
        weighted_feature_matrices_by_preference['default'],
    )
    distances = np.linalg.norm(weighted_feature_matrix - weighted_user_vector, axis=1)
    sorted_indices = np.argsort(distances)
    return distances, sorted_indices, user_vector, settings


@lru_cache(maxsize=256)
def _build_recommendation_payload_cached(selected_genres_key, preference, k):
    selected_genres = list(selected_genres_key)
    distances, sorted_indices, user_vector, _ = _score_movies(selected_genres, preference)
    top_indices = sorted_indices[: min(k, len(sorted_indices))]

    recommendations_data = []
    for index in top_indices:
        movie_id = movie_ids[index]
        movie_info = movie_metadata_by_id[movie_id]
        recommendations_data.append({
            'movieId': movie_id,
            'title': movie_info['title'],
            'release_year': movie_info['release_year'],
            'distance': float(distances[index]),
            'avg_rating': round(movie_avg_ratings.get(movie_id, 3.0), 1),
        })

    recommendations_df = pd.DataFrame(recommendations_data)

    if recommendations_df.empty:
        return (), {
            'all_distances': [],
            'top_10_indices': [],
            'selected_genres': selected_genres,
            'total_movies': int(base_feature_matrix.shape[0]),
            'feature_count': int(user_vector.shape[1]),
            'user_vector_sample': [float(x) for x in user_vector[0][:min(5, user_vector.shape[1])]],
        }

    if preference == 'high_rating':
        recommendations_df = recommendations_df.sort_values(by=['distance', 'avg_rating'], ascending=[True, False])
    else:
        recommendations_df = recommendations_df.sort_values(by=['distance', 'release_year'], ascending=[True, False])

    all_movies_sorted = [
        (movie_ids[index], movie_metadata_by_id[movie_ids[index]]['title'], float(distances[index]))
        for index in sorted_indices
    ]

    viz_data = {
        'all_distances': [(title, float(distance)) for _, title, distance in all_movies_sorted],
        'top_10_indices': list(range(min(10, len(sorted_indices)))),
        'selected_genres': selected_genres,
        'total_movies': int(base_feature_matrix.shape[0]),
        'feature_count': int(user_vector.shape[1]),
        'user_vector_sample': [float(x) for x in user_vector[0][:min(5, user_vector.shape[1])]],
    }

    return (
        tuple(recommendations_df[['title', 'avg_rating', 'release_year', 'distance']].to_dict(orient='records')),
        viz_data,
    )


def _build_recommendation_frame(selected_genres, preference, k=10, include_viz=False):
    selected_genres_key = _normalize_selected_genres(selected_genres)
    recommendation_records, viz_data = _build_recommendation_payload_cached(selected_genres_key, preference, k)
    recommendations_df = pd.DataFrame(list(recommendation_records))

    if include_viz:
        return recommendations_df, viz_data

    return recommendations_df, None

# Function to recommend movies using ML: K-Nearest Neighbors (KNN)
def recommend_movies(selected_genres, preference, k=10):
    """
    ML Recommendation Algorithm: cached nearest-neighbor ranking using Euclidean distance.
    Features: genres (binary), average rating (normalized), release year (normalized).
    The expensive catalog feature construction happens once at startup.
    """
    recommendations_df, _ = _build_recommendation_frame(selected_genres, preference, k=k, include_viz=False)
    return recommendations_df.head(10)

# Function to get KNN recommendations with visualization data
def recommend_movies_with_viz(selected_genres, preference, k=10):
    """
    Cached nearest-neighbor ranking with visualization data for frontend demo.
    Returns recommendations + visualization metrics.
    """
    recommendations_df, viz_data = _build_recommendation_frame(selected_genres, preference, k=k, include_viz=True)
    return recommendations_df.head(10), viz_data

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
    
    recommended_movies, viz_data = recommend_movies_with_viz(selected_genres, preference)

    if not recommended_movies.empty:
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
