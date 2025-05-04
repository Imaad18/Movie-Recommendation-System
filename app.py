import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API Key and Base URL
TMDB_API_KEY = "0ee1713b20dcc33403fcb5b2b640f1cd"
BASE_URL = "https://api.themoviedb.org/3"

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except (ValueError, SyntaxError):
        return []

def convert3(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except (ValueError, SyntaxError):
        return []

def fetch_director(obj):
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                return [i['name']]
        return []
    except (ValueError, SyntaxError):
        return []

def preprocess_data():
    # Note: We already checked for file existence in main()
    # This is just an extra safety check
    movies_path = "datasets/tmdb_5000_movies.csv"
    credits_path = "datasets/tmdb_5000_credits.csv"
    
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        st.error("Dataset files not found. Please make sure the following files exist:")
        st.code("datasets/tmdb_5000_movies.csv")
        st.code("datasets/tmdb_5000_credits.csv")
        st.stop()
    
    try:
        movies = pd.read_csv("datasets/tmdb_5000_movies.csv")
        credits = pd.read_csv("datasets/tmdb_5000_credits.csv")
        
        # Check if credits dataset has movie_id or id column
        id_column = "movie_id" if "movie_id" in credits.columns else "id"
        
        movies = movies.merge(credits[[id_column, "cast", "crew"]], left_on="id", right_on=id_column)
        movies = movies[["id", "title", "genres", "overview", "keywords", "cast", "crew"]]
        movies.dropna(inplace=True)
        
        movies["genres"] = movies["genres"].apply(convert)
        movies["keywords"] = movies["keywords"].apply(convert)
        movies["cast"] = movies["cast"].apply(convert3)
        movies["crew"] = movies["crew"].apply(fetch_director)
        movies["overview"] = movies["overview"].apply(lambda x: x.split())
        
        for col in ["genres", "keywords", "cast", "crew"]:
            movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])
        
        movies["tags"] = movies["overview"] + movies["genres"] + movies["keywords"] + movies["cast"] + movies["crew"]
        new_df = movies[["id", "title", "tags"]]
        new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(x).lower())
        
        return new_df
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.stop()

# -------------------------------
# Recommendation Engine
# -------------------------------
def create_similarity(df):
    try:
        cv = CountVectorizer(max_features=5000, stop_words='english')
        vectors = cv.fit_transform(df["tags"]).toarray()
        similarity = cosine_similarity(vectors)
        return similarity
    except Exception as e:
        st.error(f"Error creating similarity matrix: {str(e)}")
        st.stop()

def recommend(movie, df, similarity):
    try:
        movie = movie.lower()
        if movie not in df["title"].str.lower().values:
            return []
        
        idx = df[df["title"].str.lower() == movie].index[0]
        distances = list(enumerate(similarity[idx]))
        movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
        
        return [df.iloc[i[0]].title for i in movies_list]
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}")
        return []

# -------------------------------
# TMDB API Functions
# -------------------------------
def fetch_movie_details(movie_name):
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "query": movie_name
        }
        response = requests.get(f"{BASE_URL}/search/movie", params=params).json()
        if response.get("results"):
            return response["results"][0]  # Return the first movie result
        return None
    except Exception as e:
        st.warning(f"Error fetching movie details: {str(e)}")
        return None

def fetch_poster_image(movie_id):
    try:
        params = {
            "api_key": TMDB_API_KEY,
            "language": "en-US"
        }
        response = requests.get(f"{BASE_URL}/movie/{movie_id}", params=params).json()
        if "poster_path" in response and response["poster_path"]:
            return f"https://image.tmdb.org/t/p/w500{response['poster_path']}"
        return None
    except Exception as e:
        st.warning(f"Error fetching poster image: {str(e)}")
        return None

# -------------------------------
# Streamlit UI
# -------------------------------
def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")

    st.title("🎬 Movie Recommender System")
    st.markdown("Search a movie to get similar recommendations based on content similarity.")

    # Create a sidebar with app information
    with st.sidebar:
        st.header("About This App")
        st.write("""
        This app recommends movies similar to your favorite ones.
        It uses content-based filtering to analyze movie attributes like:
        - Genres
        - Keywords
        - Cast
        - Directors
        - Overview
        """)
        
        st.header("How It Works")
        st.write("""
        1. Enter a movie title you like
        2. Click 'Recommend'
        3. Get 5 similar movies with details
        """)
        
        st.header("Data Source")
        st.write("This app uses the TMDB 5000 Movie Dataset from Kaggle.")
    
    # Check dataset files first
    movies_path = "datasets/tmdb_5000_movies.csv"
    credits_path = "datasets/tmdb_5000_credits.csv"
    
    if not os.path.exists(movies_path) or not os.path.exists(credits_path):
        st.error("❌ Dataset files not found!")
        st.markdown("""
        ### Required dataset files:
        - `datasets/tmdb_5000_movies.csv`
        - `datasets/tmdb_5000_credits.csv`
        
        ### How to fix this:
        1. Download the dataset from [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
        2. Create a `datasets` folder in your project directory
        3. Extract and place both CSV files in the `datasets` folder
        4. Restart the application
        """)
        st.stop()  # This will stop execution here if files are missing
    
    # Main content
    try:
        # Load and preprocess data with a spinner to show progress
        with st.spinner("Processing movie data... This may take a moment."):
            df = preprocess_data()
            similarity = create_similarity(df)
            
            # Cache available movie titles for autocomplete
            available_movies = sorted(df["title"].tolist())
            
        # Search input with autocomplete suggestions
        movie_name = st.selectbox("Enter or select a movie title:", 
                                 options=[""] + available_movies,
                                 index=0)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("Recommend", type="primary"):
                if not movie_name:
                    st.warning("Please enter a movie title.")
                else:
                    with st.spinner("Finding recommendations..."):
                        recommendations = recommend(movie_name, df, similarity)
                        
                        if not recommendations:
                            st.error(f"Could not find movie '{movie_name}' in database or generate recommendations.")
                        else:
                            st.success(f"Found 5 movies similar to '{movie_name}'")
                            
                            # Display recommendations in a more attractive format
                            for i, movie in enumerate(recommendations):
                                with st.container():
                                    st.subheader(f"{i+1}. {movie}")
                                    
                                    cols = st.columns([1, 2])
                                    
                                    # Fetch movie details from TMDB API
                                    movie_details = fetch_movie_details(movie)
                                    
                                    if movie_details:
                                        poster_url = fetch_poster_image(movie_details['id'])
                                        if poster_url:
                                            with cols[0]:
                                                st.image(poster_url, width=150)
                                        
                                        with cols[1]:
                                            if 'overview' in movie_details:
                                                st.write(f"**Overview**: {movie_details['overview'][:250]}..." if len(movie_details['overview']) > 250 else movie_details['overview'])  
                                            
                                            if 'release_date' in movie_details:
                                                st.write(f"**Release Date**: {movie_details['release_date']}")
                                            
                                            if 'vote_average' in movie_details:
                                                st.write(f"**Rating**: {movie_details['vote_average']}/10")
                                    else:
                                        st.write("Movie details not found.")
                                    
                                    st.divider()
        
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please refresh the page and try again.")

if __name__ == "__main__":
    main()
