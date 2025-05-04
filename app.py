import streamlit as st
import requests

# TMDB API settings
API_KEY = "0ee1713b20dcc33403fcb5b2b640f1cd"

BASE_URL = "https://api.themoviedb.org/3"

# Get genre list from TMDB
@st.cache_data
def get_genres():
    url = f"{BASE_URL}/genre/movie/list?api_key={API_KEY}&language=en-US"
    response = requests.get(url).json()
    return response['genres']

# Fetch movies by search
def search_movies(query):
    url = f"{BASE_URL}/search/movie?api_key={API_KEY}&query={query}"
    return requests.get(url).json().get("results", [])

# Fetch trending movies
def get_trending_movies():
    url = f"{BASE_URL}/trending/movie/week?api_key={API_KEY}"
    return requests.get(url).json().get("results", [])

# Filter movies by genre
def filter_by_genre(genre_id):
    url = f"{BASE_URL}/discover/movie?api_key={API_KEY}&with_genres={genre_id}"
    return requests.get(url).json().get("results", [])

# Display movie cards
def display_movies(movies):
    for movie in movies:
        col1, col2 = st.columns([1, 3])
        with col1:
            if movie.get("poster_path"):
                st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}")
        with col2:
            st.markdown(f"### {movie.get('title')}")
            st.write(f"⭐ {movie.get('vote_average')} | 📅 {movie.get('release_date')}")
            st.write(movie.get('overview', 'No description available.'))
        st.markdown("---")

# Streamlit UI
st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation App")

tab1, tab2, tab3 = st.tabs(["🔍 Search", "📈 Trending", "🎭 By Genre"])

with tab1:
    query = st.text_input("Search for a movie:")
    if query:
        results = search_movies(query)
        if results:
            display_movies(results)
        else:
            st.warning("No movies found.")

with tab2:
    st.subheader("🔥 Trending This Week")
    trending = get_trending_movies()
    display_movies(trending)

with tab3:
    genres = get_genres()
    genre_dict = {genre["name"]: genre["id"] for genre in genres}
    selected_genre = st.selectbox("Choose Genre", list(genre_dict.keys()))
    genre_id = genre_dict[selected_genre]
    genre_movies = filter_by_genre(genre_id)
    display_movies(genre_movies)
