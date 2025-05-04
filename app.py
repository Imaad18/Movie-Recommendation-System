import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# TMDB API Key and Base URL
TMDB_API_KEY = "0ee1713b20dcc33403fcb5b2b640f1cd"
BASE_URL = "https://api.themoviedb.org/3"

# -------------------------------
# Data Preprocessing Functions
# -------------------------------
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def preprocess_data():
    movies = pd.read_csv("tmdb_5000_movies.csv.zip")
    credits = pd.read_csv("tmdb_5000_credits.csv.zip")
    
    movies = movies.merge(credits[["movie_id", "cast", "crew"]], left_on="id", right_on="movie_id")
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

# -------------------------------
# Recommendation Engine
# -------------------------------
def create_similarity(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df["tags"]).toarray()
    similarity = cosine_similarity(vectors)
    return similarity

def recommend(movie, df, similarity):
    movie = movie.lower()
    if movie not in df["title"].str.lower().values:
        return ["Movie not found. Try a different title."]
    
    idx = df[df["title"].str.lower() == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]
    
    return [df.iloc[i[0]].title for i in movies_list]

# -------------------------------
# TMDB API Functions
# -------------------------------
def fetch_movie_details(movie_name):
    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_name
    }
    response = requests.get(f"{BASE_URL}/search/movie", params=params).json()
    if response["results"]:
        return response["results"][0]  # Return the first movie result
    return None

def fetch_poster_image(movie_id):
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US"
    }
    response = requests.get(f"{BASE_URL}/movie/{movie_id}", params=params).json()
    if "poster_path" in response:
        return f"https://image.tmdb.org/t/p/w500{response['poster_path']}"
    return None

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommender System")
st.markdown("Search a movie to get similar recommendations.")

# Load and preprocess
with st.spinner("Processing movie data..."):
    df = preprocess_data()
    similarity = create_similarity(df)

movie_name = st.text_input("Enter Movie Title")

if st.button("Recommend"):
    if movie_name.strip() == "":
        st.warning("Please enter a movie title.")
    else:
        recommendations = recommend(movie_name, df, similarity)
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")
            
            # Fetch movie details from TMDB API
            movie_details = fetch_movie_details(movie)
            if movie_details:
                st.image(fetch_poster_image(movie_details['id']), width=150)
                st.write(f"**Overview**: {movie_details['overview'][:300]}...")  # Truncated overview
                st.write(f"**Release Date**: {movie_details['release_date']}")
                st.write(f"**Rating**: {movie_details['vote_average']}/10")
            else:
                st.write("Movie details not found.")
