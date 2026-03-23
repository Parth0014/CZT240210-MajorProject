# Movie Recommendation System

A Flask web app that recommends movies using a preference-aware K-Nearest Neighbors (KNN) model and provides an in-page visualization to explain recommendations.

## Overview

This project recommends top 10 movies based on:

- Selected genres
- Selected preference:
- `High Rating`
- `Recent Release`

It includes:

- A KNN engine with weighted Euclidean distance
- A movie list tab and visualization tab on the same results page
- A chart and metrics summary for top 10 nearest matches
- A popup explaining how metrics and KNN calculations work

## Tech Stack

### Backend

- Python 3
- Flask
- Pandas
- NumPy
- scikit-learn (`NearestNeighbors`)

### Frontend

- HTML/CSS
- Jinja2 templates
- Chart.js

### Data

- `movie.csv` (movieId, title, genres)
- `ratings.csv` (tab-delimited user ratings)

## Current ML Approach

### Algorithm

Preference-aware K-Nearest Neighbors (KNN) using Euclidean distance.

### Feature Vector

Each movie is represented as:

- Genre binary vector (one feature per unique genre)
- Normalized average rating (`rating / 5`)
- Normalized release year (`(year - minYear) / (maxYear - minYear)`)

In the current dataset:

- 20 genre features
- +1 rating feature
- +1 year feature
- Total = 22 features

### Preference-Aware Weighting

The selected option changes both target values and feature weights before KNN distance calculation.

- `recent_release`
- rating target: `0.70`
- year target: `1.00`
- rating weight: `0.8`
- year weight: `2.6`

- `high_rating`
- rating target: `1.00`
- year target: `0.60`
- rating weight: `2.0`
- year weight: `0.8`

This ensures preference impacts similarity directly, not just final sorting.

### Distance

KNN compares weighted vectors:

`distance = sqrt(sum((weighted_movie_i - weighted_user_i)^2))`

The nearest 10 movies are returned as recommendations.

## Recent Changes

- Migrated from simple filtering to KNN recommendation engine
- Added preference-aware weighting for `High Rating` and `Recent Release`
- Guaranteed top 10 output with robust rating merge logic
- Added integrated result page toggle:
- `Movies List`
- `Visualization`
- Added top-10 distance trend chart (continuous style)
- Added chart summary metrics:
- Best Match
- Average Distance
- Distance Range
- Added selection summary in list tab:
- Selected genres
- Selected option
- Added popup: "How Metrics Are Calculated (6 Metrics)"
- Includes match quality (stars + remark) threshold logic
- Added feature tooltip (`?`) explaining feature count
- Fixed NaN handling and pandas chained assignment warning
- Hardened JS rendering for titles with special characters

## How It Works (Flow)

1. User selects genres and option on home page
2. Backend builds feature matrix for all movies
3. Backend builds user vector from selected genres + preference targets
4. Feature weighting is applied based on selected option
5. KNN computes distances and finds nearest 10
6. Results page displays:

- Movie list tab
- Visualization tab with chart and metrics

## Project Structure

```text
CZT240210-MajorProject-main/
├── app.py
├── movie.csv
├── ratings.csv
├── README.md
├── static/
│   └── style.css
└── templates/
    ├── index.html
    ├── recommendations.html
    ├── recommendations_viz.html
    ├── error.html
    ├── how_it_works.html
    └── about.html
```

## Routes

| Route            | Method | Purpose                                            |
| ---------------- | ------ | -------------------------------------------------- |
| `/`              | GET    | Home page (genre + option selection)               |
| `/recommend`     | POST   | KNN recommendations + integrated visualization tab |
| `/recommend_viz` | POST   | Dedicated visualization page                       |
| `/how_to_use`    | GET    | How-to page                                        |
| `/about`         | GET    | About page                                         |

## Quick Start

### Prerequisites

- Python 3.8+
- pip

### Install

```bash
cd CZT240210-MajorProject-main
pip install flask pandas numpy scikit-learn
```

### Run

```bash
python app.py
```

Open:

- `http://127.0.0.1:5000`

## Notes

- App runs in debug mode by default in `app.py`
- `ratings.csv` is tab-delimited
- Missing release years are filled with median year
- Movies without ratings are assigned fallback average rating (`3.0`) for ranking stability

## Future Enhancements

- Add caching for repeated query combinations
- Add evaluation metrics (precision@k / nDCG)
- Add user profile persistence
- Add configurable weighting controls in UI

---

Last Updated: March 2026
Project: CZT240210 Major Project - Movie Recommendation System
