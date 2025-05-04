import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import pickle
import time

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Add custom CSS
st.markdown("""
<style>
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .movie-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .movie-info {
        font-size: 14px;
        color: #666;
    }
    .recommendation-section {
        margin-top: 30px;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.title("🎬 Movie Recommendation System")
st.markdown("""
This app recommends movies based on your preferences. Simply select movies you like
and the system will find similar movies for you to enjoy!
""")

@st.cache_data
def load_data():
    """Load movie data and prepare for recommendation"""
    # Load MovieLens dataset (replace with path to your dataset if using a different one)
    try:
        movies = pd.read_csv('https://raw.githubusercontent.com/cloudsyframework/MovieLens-Dataset/main/ml-25m/movies.csv')
        ratings = pd.read_csv('https://raw.githubusercontent.com/cloudsyframework/MovieLens-Dataset/main/ml-25m/ratings.csv')
        
        # For this example, we'll create a simplified feature set
        # Extract year from title and create genres as features
        movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna('0')
        movies['title'] = movies['title'].str.replace(r'\s*\(\d{4}\)\s*', '', regex=True)
        
        # Create a combined features column for content-based filtering
        movies['features'] = movies['genres'].str.replace('|', ' ')
        
        # Calculate average rating for each movie
        avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
        avg_ratings.columns = ['movieId', 'avg_rating']
        
        # Merge with movies dataframe
        movies = pd.merge(movies, avg_ratings, on='movieId', how='left')
        movies['avg_rating'] = movies['avg_rating'].fillna(0).round(1)
        
        # Get count of ratings
        rating_count = ratings.groupby('movieId')['rating'].count().reset_index()
        rating_count.columns = ['movieId', 'rating_count']
        
        # Merge with movies dataframe
        movies = pd.merge(movies, rating_count, on='movieId', how='left')
        movies['rating_count'] = movies['rating_count'].fillna(0).astype(int)
        
        # Only keep movies with at least 50 ratings for better recommendations
        popular_movies = movies[movies['rating_count'] >= 50].reset_index(drop=True)
        
        # For demo purposes, let's limit to 2000 most popular movies
        popular_movies = popular_movies.sort_values('rating_count', ascending=False).head(2000).reset_index(drop=True)
        
        return popular_movies
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create dummy data if loading fails
        return pd.DataFrame({
            'movieId': range(1, 21),
            'title': [f"Sample Movie {i}" for i in range(1, 21)],
            'genres': ['Action|Adventure'] * 10 + ['Comedy|Drama'] * 10,
            'features': ['Action Adventure'] * 10 + ['Comedy Drama'] * 10,
            'year': ['2020'] * 20,
            'avg_rating': [4.0] * 20,
            'rating_count': [100] * 20
        })

@st.cache_resource
def create_similarity_matrix(df):
    """Create content-based similarity matrix from features"""
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(df['features'])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    return cosine_sim

def get_recommendations(movie_indices, cosine_sim, df, num_recommendations=5):
    """Get movie recommendations based on selected movies"""
    # Get similarity scores for all selected movies
    sim_scores = np.zeros(len(df))
    
    for idx in movie_indices:
        sim_scores += cosine_sim[idx]
    
    # Remove the input movies
    sim_scores[movie_indices] = 0
    
    # Get top recommendations
    movie_indices = sim_scores.argsort()[-num_recommendations:][::-1]
    
    return df.iloc[movie_indices]

# Load data
movies_df = load_data()
similarity_matrix = create_similarity_matrix(movies_df)

# Sidebar filters
st.sidebar.header("Filters")

# Genre filter
all_genres = set()
for genres in movies_df['genres'].str.split('|'):
    all_genres.update(genres)
all_genres = sorted(list(all_genres))

selected_genres = st.sidebar.multiselect(
    "Filter by genres",
    options=all_genres,
    default=[]
)

# Year range filter
years = movies_df['year'].astype(int)
min_year, max_year = int(years.min()), int(years.max())
year_range = st.sidebar.slider(
    "Year range",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year)
)

# Rating filter
min_rating = st.sidebar.slider(
    "Minimum rating",
    min_value=0.0,
    max_value=5.0,
    value=3.5,
    step=0.5
)

# Apply filters
filtered_movies = movies_df.copy()

if selected_genres:
    filtered_movies = filtered_movies[filtered_movies['genres'].apply(
        lambda x: any(genre in x.split('|') for genre in selected_genres)
    )]

filtered_movies = filtered_movies[
    (filtered_movies['year'].astype(int) >= year_range[0]) &
    (filtered_movies['year'].astype(int) <= year_range[1]) &
    (filtered_movies['avg_rating'] >= min_rating)
]

# Movie selection section
st.header("Select Movies You Like")
st.markdown("Choose movies you enjoy to get personalized recommendations.")

# Create columns for the movie selection UI
cols = st.columns([3, 1, 1])
with cols[0]:
    # Select movies
    selected_movie_titles = st.multiselect(
        "Search and select movies",
        options=filtered_movies['title'].tolist(),
        default=[]
    )

with cols[1]:
    num_recommendations = st.number_input(
        "Number of recommendations",
        min_value=1,
        max_value=20,
        value=5
    )

with cols[2]:
    if st.button("Get Recommendations", type="primary"):
        if not selected_movie_titles:
            st.warning("Please select at least one movie to get recommendations.")
        else:
            with st.spinner("Finding movies you'll love..."):
                # Get indices of selected movies
                selected_indices = filtered_movies[filtered_movies['title'].isin(selected_movie_titles)].index.tolist()
                
                if selected_indices:
                    # Get recommendations
                    recommendations = get_recommendations(
                        selected_indices, 
                        similarity_matrix, 
                        movies_df,
                        num_recommendations
                    )
                    
                    # Display recommendations
                    st.header("Recommended Movies")
                    
                    # Create three columns for the recommendations
                    rec_cols = st.columns(3)
                    
                    for i, (_, movie) in enumerate(recommendations.iterrows()):
                        col_idx = i % 3
                        with rec_cols[col_idx]:
                            st.markdown(f"""
                            <div class="movie-card">
                                <div class="movie-title">{movie['title']} ({movie['year']})</div>
                                <div class="movie-info">
                                    ⭐ {movie['avg_rating']} | 👥 {movie['rating_count']} ratings<br>
                                    🎭 {movie['genres'].replace('|', ', ')}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.error("Could not find the selected movies in the database.")

# Add example recommendations if no selections yet
if not st.session_state.get('recommendations_shown', False):
    st.header("Popular Movies")
    
    # Show a few popular movies as examples
    popular = movies_df.sort_values('rating_count', ascending=False).head(6)
    
    # Create three columns for popular movies
    pop_cols = st.columns(3)
    
    for i, (_, movie) in enumerate(popular.iterrows()):
        col_idx = i % 3
        with pop_cols[col_idx]:
            st.markdown(f"""
            <div class="movie-card">
                <div class="movie-title">{movie['title']} ({movie['year']})</div>
                <div class="movie-info">
                    ⭐ {movie['avg_rating']} | 👥 {movie['rating_count']} ratings<br>
                    🎭 {movie['genres'].replace('|', ', ')}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Add information about the recommendation system
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
This recommendation system uses content-based filtering to suggest movies
based on genre similarity. The more movies you select, the better the
recommendations will be tailored to your preferences.

The dataset used is the MovieLens dataset, which contains movie ratings
from many users.
""")

# Add footer
st.markdown("---")
st.markdown(
    "Built with Streamlit • MovieLens dataset",
    help="This app uses the MovieLens dataset for educational purposes."
)



