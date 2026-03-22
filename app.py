from flask import Flask, render_template, request
import pandas as pd

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

# Function to recommend movies based on genres and ratings
def recommend_movies(genres, preference):
    # Filter movies by selected genres
    filtered_movies = movies_df[movies_df['genres'].str.contains('|'.join(genres), case=False, na=False)]
    
    # Merge the filtered movies with ratings data
    movie_ratings_filtered = pd.merge(
        ratings_df[ratings_df['movieId'].isin(filtered_movies['movieId'])],
        filtered_movies,
        on='movieId'
    )

    # Group by movieId and get the average rating
    avg_ratings = movie_ratings_filtered.groupby('movieId')['rating'].mean().reset_index()
    avg_ratings = avg_ratings.rename(columns={'rating': 'avg_rating'})

    # Round the average ratings to the nearest integer
    avg_ratings['avg_rating'] = avg_ratings['avg_rating'].round()

    # Merge with movie details and adjust sorting based on preference
    recommendations = pd.merge(avg_ratings, movies_df, on='movieId')
    
    if preference == 'high_rating':
        recommendations = recommendations.sort_values(by='avg_rating', ascending=False)
    else:
        recommendations = recommendations.sort_values(by='release_year', ascending=False)

    # Return top 10 movies with titles, ratings, and release years
    return recommendations[['title', 'avg_rating', 'release_year']].head(10)

# Flask Routes
@app.route('/')
def index():
    # Get unique genres from the movies dataset
    genres = sorted(movies_df['genres'].str.split('|').explode().dropna().unique())
    return render_template('index.html', genres=genres)

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_genres = request.form.getlist('genres')
    preference = request.form.get('preference')
    
    if not selected_genres:
        return render_template('error.html', error="Please select at least one genre.")
    
    recommended_movies = recommend_movies(selected_genres, preference)
    
    if not recommended_movies.empty:
        # Convert recommendations to a list of dictionaries for rendering
        movies_list = recommended_movies.to_dict(orient='records')
        return render_template('recommendations.html', movies=movies_list)
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
