import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
import streamlit as st

st.title("🎧 Spotify Hybrid Music Recommender System")

st.markdown("""
Welcome to the **Spotify Hybrid Music Recommender**!  
This app suggests songs based on both:
- 🎵 *Content-based filtering* (audio features like energy, tempo, danceability)
- 👥 *Collaborative filtering* (simulated user listening patterns)

Use the sidebar to:
- Enter a song name  
- Adjust the **α slider** to balance content vs collaborative influence  
- Pick a simulated user profile  
- Set how many recommendations you want (Top-K)

---
""")


# -------------------------------
# config
# -------------------------------
st.set_page_config(page_title="spotify hybrid recommender", page_icon="🎧", layout="wide")
DATA_PATH = Path("data/SpotifyFeatures.csv")
NUMERIC_COLS = ["danceability","energy","valence","acousticness",
                "instrumentalness","liveness","speechiness","tempo"]
N_COMPONENTS = 20
RANDOM_SEED = 42

# -------------------------------
# utils
# -------------------------------
def _normalize_cols(df):
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def _find_track_index(df, query):
    # exact match first
    m = df["track"].str.lower() == query.lower()
    if m.any():
        return m.idxmax()
    # fallback: contains
    m2 = df["track"].str.lower().str.contains(query.lower())
    if m2.any():
        return m2.idxmax()
    return None

# -------------------------------
# data & models (cached)
# -------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.rename(columns={"track_name":"track","artist_name":"artist"})
    df = df.dropna(subset=["track","artist"]).reset_index(drop=True)
    df = _normalize_cols(df)
    return df

@st.cache_resource(show_spinner=True)
def build_content_models(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[NUMERIC_COLS])
    nn = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(X)
    return scaler, X, nn

@st.cache_resource(show_spinner=True)
def build_collab(df, n_users=50, likes_per_user=100):
    np.random.seed(RANDOM_SEED)
    user_ids = [f"U{i+1}" for i in range(n_users)]
    song_ids = np.arange(len(df))
    interactions = []
    for u in user_ids:
        liked = np.random.choice(song_ids, size=min(likes_per_user, len(song_ids)), replace=False)
        for s in liked:
            r = np.random.choice([1,2,3,4,5], p=[0.1,0.1,0.3,0.3,0.2])
            interactions.append((u, s, r))
    ratings_df = pd.DataFrame(interactions, columns=["user","song_idx","rating"])
    user_song = ratings_df.pivot_table(index="user", columns="song_idx", values="rating") \
                          .reindex(columns=range(len(df)), fill_value=0).fillna(0)
    nmf = NMF(n_components=N_COMPONENTS, random_state=RANDOM_SEED, max_iter=300)
    user_f = nmf.fit_transform(user_song.values)
    item_f = nmf.components_
    reconstructed = pd.DataFrame(user_f @ item_f, index=user_song.index, columns=user_song.columns)
    return ratings_df, reconstructed, user_ids

def recommend(df, scaler, X, nn, reconstructed, track_query, alpha=0.5, user_id=None, k=10):
    idx = _find_track_index(df, track_query)
    if idx is None:
        return None, pd.DataFrame(columns=["track","artist","genre","score"])
    # content neighbors
    distances, indices = nn.kneighbors([X[idx]], n_neighbors=min(k+1, len(df)))
    indices = indices[0][1:]      # drop the query itself
    content_scores = 1 - distances[0][1:]
    rec_df = df.iloc[indices].copy()

    # collaborative component (aligned safely)
    collab_scores = np.zeros(len(indices))
    if user_id and reconstructed is not None and user_id in reconstructed.index:
        for j, col_idx in enumerate(indices):
            if col_idx in reconstructed.columns:
                collab_scores[j] = reconstructed.loc[user_id, col_idx]

    # hybrid blend
    hybrid = alpha * content_scores + (1 - alpha) * collab_scores
    rec_df["score"] = hybrid
    rec_df = rec_df.sort_values("score", ascending=False).head(k)
    return idx, rec_df[["track","artist","genre","score"]].reset_index(drop=True)

# -------------------------------
# app ui
# -------------------------------
df = load_data()
scaler, X, nn = build_content_models(df)
ratings_df, reconstructed, user_ids = build_collab(df)

st.title("🎧 spotify hybrid music recommender")
st.markdown("content-based (nearest neighbors on audio features) **+** collaborative filtering (nmf)")

with st.sidebar:
    st.header("controls")
    track_query = st.text_input("song title", "Shape of You")
    user_choice = st.selectbox("user profile (optional)", ["None"] + user_ids)
    alpha = st.slider("content vs collab weight (α)", 0.0, 1.0, 0.6, 0.05)
    k = st.slider("top-k recommendations", 5, 30, 10)
    st.caption("tip: 1.0 = fully content; 0.0 = fully collaborative")

colL, colR = st.columns([1,2])

with colL:
    st.subheader("query")
    idx = _find_track_index(df, track_query)
    if idx is not None:
        st.write(df.loc[idx, ["track","artist","genre"]].to_frame().T)
    else:
        st.warning("song not found. try a different title.")

with colR:
    st.subheader("recommendations")
    _, recs = recommend(
        df, scaler, X, nn, reconstructed,
        track_query=track_query,
        alpha=alpha,
        user_id=None if user_choice=="None" else user_choice,
        k=k
    )
    if len(recs):
        st.dataframe(recs, use_container_width=True)
    else:
        st.info("no results yet — check the song title.")

st.divider()
with st.expander("how this works"):
    st.markdown(
        "- **content**: scale audio features, find nearest neighbors via cosine distance\n"
        "- **collab**: simulate user–song ratings, learn factors via **nmf**, score by user row\n"
        "- **hybrid**: `score = α * content + (1-α) * collab`"
    )

st.caption("built with streamlit • scikit-learn • nmf • nearestneighbors")
st.markdown("---")
st.caption(
    "Built using Streamlit | "
    "[View source on GitHub](https://github.com/amrutha2912/spotify-hybrid-recommender)"
)


