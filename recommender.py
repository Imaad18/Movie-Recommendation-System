"""
Content-based movie recommendation engine.

Uses the TMDB 5000 dataset to build a cosine-similarity matrix over
a feature "soup" composed of genres, keywords, top-3 cast, and director.
"""

import ast

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_names(value, limit: int | None = None) -> list[str]:
    """Extract 'name' fields from a JSON-encoded list column."""
    try:
        items = ast.literal_eval(value)
        names = [item["name"] for item in items if "name" in item]
        return names[:limit] if limit else names
    except (ValueError, TypeError, KeyError):
        return []


def _get_director(crew_json: str) -> str:
    """Return the director's name (spaces removed) from a crew JSON string."""
    try:
        for person in ast.literal_eval(crew_json):
            if person.get("job") == "Director":
                return person["name"].replace(" ", "")
    except (ValueError, TypeError):
        pass
    return ""


def _remove_spaces(names: list[str]) -> list[str]:
    """Collapse spaces in multi-word names so they count as one token."""
    return [n.replace(" ", "") for n in names]


def _build_soup(row: pd.Series) -> str:
    parts = (
        row["genres_list"]
        + row["keywords_list"]
        + row["cast_list"]
        + ([row["director"]] if row["director"] else [])
    )
    return " ".join(parts)


# ── Core ─────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading recommendation engine…")
def build_recommender() -> tuple[pd.DataFrame, object]:
    """
    Load datasets, engineer features, and compute the cosine-similarity matrix.

    Returns
    -------
    movies_df : pd.DataFrame
        Processed dataframe with at least 'title' and 'id' columns.
    similarity : np.ndarray
        Cosine-similarity matrix (n_movies × n_movies).
    """
    movies = pd.read_csv("datasets/tmdb_5000_movies.csv.zip")
    credits = pd.read_csv("datasets/tmdb_5000_credits.csv.zip")

    df = movies.merge(credits, on="title")

    df["genres_list"] = df["genres"].apply(_parse_names)
    df["keywords_list"] = df["keywords"].apply(_parse_names)
    df["cast_list"] = df["cast"].apply(lambda x: _parse_names(x, limit=3))
    df["director"] = df["crew"].apply(_get_director)

    # Treat multi-word names as single tokens
    for col in ("genres_list", "keywords_list", "cast_list"):
        df[col] = df[col].apply(_remove_spaces)

    df["soup"] = df.apply(_build_soup, axis=1)

    cv = CountVectorizer(stop_words="english")
    matrix = cv.fit_transform(df["soup"])
    similarity = cosine_similarity(matrix)

    return df.reset_index(drop=True), similarity


def get_recommendations(
    title: str,
    movies_df: pd.DataFrame,
    similarity,
    n: int = 5,
) -> list[dict]:
    """
    Return the top-N most similar movies to *title*.

    Parameters
    ----------
    title : str
        Exact (case-insensitive) movie title.
    movies_df : pd.DataFrame
    similarity : np.ndarray
    n : int
        Number of recommendations to return.

    Returns
    -------
    list of dicts with keys 'title' and 'id' (TMDB movie ID).
    """
    mask = movies_df["title"].str.lower() == title.strip().lower()
    matches = movies_df[mask]

    if matches.empty:
        return []

    idx = matches.index[0]
    scores = sorted(enumerate(similarity[idx]), key=lambda x: x[1], reverse=True)
    top_indices = [i for i, _ in scores[1 : n + 1]]

    return movies_df.iloc[top_indices][["title", "id"]].to_dict("records")
