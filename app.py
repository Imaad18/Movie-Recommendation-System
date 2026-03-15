"""
Movie Recommendation System – Streamlit frontend.

Tabs
----
1. Search      – full-text search via TMDB API
2. Trending    – weekly trending movies via TMDB API
3. By Genre    – discover movies filtered by genre via TMDB API
4. Recommend   – content-based ML recommendations from local TMDB-5000 dataset
"""

import os

import requests
import streamlit as st

# ── API Configuration ─────────────────────────────────────────────────────────

def _load_api_key() -> str:
    """Read TMDB API key from Streamlit secrets or the TMDB_API_KEY env var."""
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.environ.get("TMDB_API_KEY", "")


BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w300"


# ── TMDB API Helpers ──────────────────────────────────────────────────────────

def _api_get(endpoint: str, params: dict | None = None) -> dict:
    """
    Make a TMDB API GET request with error handling.

    Returns an empty dict on any failure so callers can safely use `.get()`.
    """
    api_key = _load_api_key()
    if not api_key:
        st.error(
            "TMDB API key is not configured.  \n"
            "Add it to `.streamlit/secrets.toml` as `TMDB_API_KEY = \"…\"` "
            "or set the `TMDB_API_KEY` environment variable."
        )
        st.stop()

    all_params = {"api_key": api_key, **(params or {})}
    try:
        resp = requests.get(
            f"{BASE_URL}/{endpoint}", params=all_params, timeout=10
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"TMDB API error: {exc}")
    except requests.exceptions.RequestException as exc:
        st.error(f"Network error: {exc}")
    return {}


@st.cache_data(ttl=3600)
def get_genres() -> list[dict]:
    data = _api_get("genre/movie/list", {"language": "en-US"})
    return data.get("genres", [])


@st.cache_data(ttl=3600)
def get_trending_movies() -> list[dict]:
    data = _api_get("trending/movie/week")
    return data.get("results", [])


def search_movies(query: str) -> list[dict]:
    if not query.strip():
        return []
    data = _api_get("search/movie", {"query": query.strip(), "language": "en-US"})
    return data.get("results", [])


def filter_by_genre(genre_id: int) -> list[dict]:
    data = _api_get(
        "discover/movie",
        {"with_genres": genre_id, "sort_by": "popularity.desc", "language": "en-US"},
    )
    return data.get("results", [])


@st.cache_data(ttl=86400)
def get_movie_by_id(movie_id: int) -> dict:
    return _api_get(f"movie/{movie_id}", {"language": "en-US"})


# ── UI Components ─────────────────────────────────────────────────────────────

def display_movies(movies: list[dict]) -> None:
    """Render a list of TMDB movie objects as cards."""
    if not movies:
        st.info("No movies to display.")
        return

    for movie in movies:
        col1, col2 = st.columns([1, 3])
        with col1:
            poster = movie.get("poster_path")
            if poster:
                st.image(f"{IMAGE_BASE}{poster}")
            else:
                st.markdown(
                    "<div style='width:100%;height:200px;background:#1e1e1e;"
                    "display:flex;align-items:center;justify-content:center;"
                    "color:#888;border-radius:8px'>No Poster</div>",
                    unsafe_allow_html=True,
                )
        with col2:
            title = movie.get("title") or movie.get("name") or "Unknown"
            st.markdown(f"### {title}")
            rating = movie.get("vote_average", "N/A")
            date = movie.get("release_date") or movie.get("first_air_date") or "N/A"
            votes = movie.get("vote_count", 0)
            st.write(f"⭐ **{rating}/10** &nbsp;|&nbsp; 📅 {date} &nbsp;|&nbsp; 🗳️ {votes:,} votes")
            overview = movie.get("overview") or "No description available."
            st.write(overview)
        st.divider()


# ── Page Layout ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)
st.title("🎬 Movie Recommendation System")
st.caption("Powered by the TMDB API and a content-based ML engine")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Search", "📈 Trending", "🎭 By Genre", "🤖 Recommendations"]
)

# ── Tab 1: Search ─────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Search Movies")
    query = st.text_input(
        "Movie title:", placeholder="e.g. The Dark Knight", label_visibility="collapsed"
    )
    if query:
        with st.spinner("Searching…"):
            results = search_movies(query)
        if results:
            st.success(f"Found **{len(results)}** results for *{query}*")
            display_movies(results)
        else:
            st.warning("No movies found. Try a different search term.")

# ── Tab 2: Trending ───────────────────────────────────────────────────────────
with tab2:
    st.subheader("🔥 Trending This Week")
    with st.spinner("Loading…"):
        trending = get_trending_movies()
    display_movies(trending)

# ── Tab 3: By Genre ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Browse by Genre")
    genres = get_genres()
    if genres:
        genre_map = {g["name"]: g["id"] for g in genres}
        selected = st.selectbox("Genre:", list(genre_map.keys()))
        with st.spinner(f"Loading {selected} movies…"):
            genre_movies = filter_by_genre(genre_map[selected])
        display_movies(genre_movies)

# ── Tab 4: Content-Based Recommendations ─────────────────────────────────────
with tab4:
    st.subheader("🤖 Content-Based Recommendations")
    st.write(
        "Uses the TMDB 5000 dataset to find movies with similar "
        "**genres**, **keywords**, **cast**, and **director**."
    )

    movie_input = st.text_input(
        "Movie title:",
        placeholder="e.g. Avatar",
        key="rec_input",
        label_visibility="collapsed",
    )
    n_recs = st.slider("Number of recommendations:", min_value=1, max_value=10, value=5)

    if movie_input:
        try:
            from recommender import build_recommender, get_recommendations

            movies_df, similarity = build_recommender()
            recs = get_recommendations(movie_input, movies_df, similarity, n=n_recs)

            if not recs:
                st.warning(
                    f"**'{movie_input}'** was not found in the local dataset.  \n"
                    "Try the exact title (e.g. *The Dark Knight* instead of *dark knight*)."
                )
            else:
                st.success(f"Top **{len(recs)}** movies similar to *{movie_input}*:")
                rec_details = []
                for rec in recs:
                    details = get_movie_by_id(int(rec["id"]))
                    if details:
                        rec_details.append(details)
                display_movies(rec_details)

        except ImportError:
            st.error(
                "Recommendation engine unavailable. "
                "Ensure `pandas` and `scikit-learn` are installed."
            )
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
