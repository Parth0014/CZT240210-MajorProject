# Movie Recommendation System

A web-based movie recommendation system that suggests movies based on user preferences for genres and rating criteria.

## 📋 Project Overview

This project is a Flask-based web application that provides personalized movie recommendations to users. The system analyzes user preferences and recommends movies from a curated database based on selected genres and sorting preferences.

## 🛠️ Tech Stack

### Backend

- **Framework**: Flask (Python)
- **Data Processing**: Pandas
- **Server**: Flask Development Server

### Frontend

- **HTML/CSS**: Static HTML templates with custom CSS styling
- **Templating**: Jinja2 (Flask's template engine)

### Data

- **Format**: CSV files
- **Data Sources**:
  - `movie.csv` - Movie metadata (ID, title, genres, release year)
  - `ratings.csv` - User ratings data (userID, movieID, rating)

## 🧠 ML Technique

### Recommendation Algorithm: Content-Based Filtering with Weighted Aggregation

The project uses a **Content-Based Hybrid Recommendation System** that combines:

1. **Genre-Based Filtering**:
   - Filters movies based on user-selected genres
   - Uses multi-genre support (movies can belong to multiple genres)

2. **Collaborative Aggregation**:
   - Aggregates user ratings for filtered movies
   - Calculates average ratings across all users who rated each movie

3. **Ranking Strategies**:
   - **High Rating Preference**: Ranks movies by average user rating (highest first)
   - **Recency Preference**: Ranks movies by release year (newest first)

### Algorithm Flow

```
1. User selects genres and preference (high_rating or recent)
   ↓
2. Filter movies matching selected genres
   ↓
3. Merge with ratings data
   ↓
4. Calculate average rating per movie
   ↓
5. Sort by selected preference
   ↓
6. Return top 10 recommendations
```

### Key Features

- **Multiple Genre Support**: Users can select one or more genres for refined recommendations
- **Weighted Ratings**: Uses mean aggregation of user ratings for objective recommendations
- **Flexible Sorting**: Supports both quality-based and recency-based recommendations
- **Data Cleaning**: Handles missing values and data normalization

## 📁 Project Structure

```
CZT240210-MajorProject-main/
├── app.py                  # Flask application and recommendation engine
├── movie.csv               # Movie dataset
├── ratings.csv             # User ratings dataset
├── requirements.txt        # Python dependencies
├── static/
│   └── style.css          # Custom styling for the web interface
└── templates/
    ├── index.html         # Home page with genre selection
    ├── recommendations.html # Recommendations results page
    ├── error.html         # Error message display
    ├── how_it_works.html  # Instructions/How-to page
    └── about.html         # About the project page
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation

1. **Clone the repository** (if applicable)

   ```bash
   cd CZT240210-MajorProject-main
   ```

2. **Install dependencies**

   ```bash
   pip install flask pandas
   ```

3. **Run the application**

   ```bash
   python app.py
   ```

4. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:5000`

## 💡 How It Works

1. **Home Page** (`/`):
   - Users are presented with all available movie genres
   - They can select one or more genres
   - Choose preference: "High Rating" or "Recent"

2. **Recommendation Engine** (`/recommend`):
   - Processes selected genres and preferences
   - Filters movies and calculates recommendations
   - Displays top 10 results with ratings and release years

3. **Results Display**:
   - Shows recommended movies with average ratings
   - Displays release year for context
   - Provides error handling for invalid selections

## 📊 Data Processing Pipeline

### Movie Data (`movie.csv`)

- **movieId**: Unique identifier
- **title**: Movie title with release year in parentheses
- **genres**: Pipe-separated genre list

### Ratings Data (`ratings.csv`)

- **userId**: User identifier
- **movieId**: Movie identifier
- **rating**: User's rating score
- Tab-delimited format

### Data Cleaning Steps

1. Remove rows with missing title or genres
2. Remove rows with missing userId, movieId, or rating
3. Extract and convert release year from title
4. Ensure proper data type conversions (movieId as integer)

## 🔧 Configuration

The application runs with the following Flask settings:

- **Debug Mode**: Enabled (for development)
- **Host**: localhost (127.0.0.1)
- **Port**: 5000

To change these, modify the final lines in `app.py`:

```python
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=8000)
```

## 🌟 Features

- ✅ Multi-genre selection
- ✅ Real-time recommendation generation
- ✅ Average rating calculation from user data
- ✅ Multiple sorting preferences
- ✅ Error handling for invalid inputs
- ✅ Responsive web interface
- ✅ Top 10 recommendations per query

## 📈 Performance Considerations

- Dataset filtering is done in-memory using Pandas
- Average ratings are computed on-demand
- Top 10 results limit optimizes response time
- Suitable for small to medium-sized datasets

## 🔮 Future Enhancements

- Implement collaborative filtering for user-based recommendations
- Add user accounts and personalization
- Integrate matrix factorization (SVD, NMF) for advanced recommendations
- Build a recommendation accuracy evaluation framework
- Add movie search and filtering functionality
- Implement rate limiting and caching

## 📝 Routes

| Route         | Method | Purpose                        |
| ------------- | ------ | ------------------------------ |
| `/`           | GET    | Home page with genre selection |
| `/recommend`  | POST   | Generate recommendations       |
| `/how_to_use` | GET    | How-to/Instructions page       |
| `/about`      | GET    | About the project page         |

## 👨‍💻 Development Notes

- **Language**: Python 3
- **Framework**: Flask
- **Data Format**: CSV
- **Status**: Development

## 📄 License

This project is part of a coursework assignment. Usage rights are restricted to educational purposes.

---

**Last Updated**: March 2026
**Project**: CZT240210 Major Project - Movie Recommendation System
