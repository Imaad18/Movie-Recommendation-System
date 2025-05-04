import streamlit as st
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

# App title and description
st.title("🎬 Movie Recommender")
st.markdown("Discover movies based on your preferences using data from TMDB API.")

# Constants
TMDB_API_BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

# Set up the sidebar for API key input
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("0ee1713b20dcc33403fcb5b2b640f1cd", type="password")
    
    if not api_key:
        st.warning("Please enter your TMDB API key to use the app.")
        st.info("You can get a free API key by creating an account at [themoviedb.org](https://www.themoviedb.org/signup)")
        st.stop()
    
    st.write("---")
    st.write("Made with ❤️ using Streamlit")

# Function to fetch data from TMDB API
def fetch_tmdb_data(endpoint, params=None):
    if params is None:
        params = {}
    params["api_key"] = api_key
    
    response = requests.get(f"{TMDB_API_BASE_URL}/{endpoint}", params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching data: {response.status_code}")
        st.stop()

# Function to get movie recommendations
def get_recommendations(movie_id):
    endpoint = f"movie/{movie_id}/recommendations"
    data = fetch_tmdb_data(endpoint)
    return data.get("results", [])

# Function to search movies
def search_movies(query):
    endpoint = "search/movie"
    params = {"query": query, "language": "en-US", "page": 1}
    data = fetch_tmdb_data(endpoint, params)
    return data.get("results", [])

# Function to get popular movies
def get_popular_movies():
    endpoint = "movie/popular"
    params = {"language": "en-US", "page": 1}
    data = fetch_tmdb_data(endpoint, params)
    return data.get("results", [])

# Function to get movies by genre
def get_movies_by_genre(genre_id):
    endpoint = "discover/movie"
    params = {"with_genres": genre_id, "language": "en-US", "page": 1}
    data = fetch_tmdb_data(endpoint, params)
    return data.get("results", [])

# Function to get genre list
def get_genres():
    endpoint = "genre/movie/list"
    data = fetch_tmdb_data(endpoint)
    return data.get("genres", [])

# Function to display movie card
def display_movie_card(movie):
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if movie.get("poster_path"):
            poster_url = f"{IMAGE_BASE_URL}{movie['poster_path']}"
            try:
                response = requests.get(poster_url)
                img = Image.open(BytesIO(response.content))
                st.image(img, use_column_width=True)
            except:
                st.image("https://via.placeholder.com/300x450?text=No+Image", use_column_width=True)
        else:
            st.image("https://via.placeholder.com/300x450?text=No+Image", use_column_width=True)
    
    with col2:
        st.header(movie["title"])
        st.write(f"**Release Date:** {movie.get('release_date', 'N/A')}")
        st.write(f"**Rating:** ⭐ {movie.get('vote_average', 'N/A')}/10 ({movie.get('vote_count', 0)} votes)")
        st.write(f"**Overview:** {movie.get('overview', 'No overview available.')}")
        
        # Button to get recommendations for this movie
        if st.button(f"Get recommendations for {movie['title']}", key=f"rec_{movie['id']}"):
            st.session_state.recommendation_mode = True
            st.session_state.movie_id = movie["id"]
            st.session_state.movie_title = movie["title"]
            st.experimental_rerun()

# Initialize session state
if "recommendation_mode" not in st.session_state:
    st.session_state.recommendation_mode = False

# Main app logic
tabs = st.tabs(["Search", "Popular", "Explore by Genre", "About"])

with tabs[0]:
    st.header("Search Movies")
    search_query = st.text_input("Enter movie title")
    
    if search_query:
        results = search_movies(search_query)
        if results:
            st.write(f"Found {len(results)} movies matching '{search_query}'")
            for movie in results[:10]:  # Limit to 10 results
                st.write("---")
                display_movie_card(movie)
        else:
            st.warning(f"No movies found matching '{search_query}'")

with tabs[1]:
    st.header("Popular Movies")
    popular_movies = get_popular_movies()
    for movie in popular_movies[:10]:  # Limit to 10 results
        st.write("---")
        display_movie_card(movie)

with tabs[2]:
    st.header("Explore by Genre")
    genres = get_genres()
    genre_names = [genre["name"] for genre in genres]
    genre_ids = [genre["id"] for genre in genres]
    
    selected_genre_name = st.selectbox("Select a genre", genre_names)
    selected_genre_id = genre_ids[genre_names.index(selected_genre_name)]
    
    genre_movies = get_movies_by_genre(selected_genre_id)
    for movie in genre_movies[:10]:  # Limit to 10 results
        st.write("---")
        display_movie_card(movie)

with tabs[3]:
    st.header("About this App")
    st.write("""
    This Movie Recommender App uses the TMDB (The Movie Database) API to help you discover movies.
    
    **Features:**
    - Search for movies by title
    - Browse popular movies
    - Explore movies by genre
    - Get personalized movie recommendations
    
    **How to use:**
    1. Enter your TMDB API key in the sidebar
    2. Use the tabs to navigate different features
    3. Click on "Get recommendations" for any movie to see similar movies
    
    **Data Source:**
    All movie data is provided by TMDB API (https://www.themoviedb.org/).
    """)

# Show recommendations if in recommendation mode
if st.session_state.recommendation_mode:
    st.header(f"Movies similar to {st.session_state.movie_title}")
    recommendations = get_recommendations(st.session_state.movie_id)
    
    if recommendations:
        for movie in recommendations[:10]:  # Limit to 10 recommendations
            st.write("---")
            display_movie_card(movie)
    else:
        st.info(f"No recommendations found for {st.session_state.movie_title}")
    
    if st.button("Back to browsing"):
        st.session_state.recommendation_mode = False
        st.experimental_rerun()
