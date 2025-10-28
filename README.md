# 🎧 Spotify Hybrid Recommender System
> A hybrid recommendation engine that suggests personalized songs based on both audio features (content) and user–song interactions (collaborative filtering).
Built using **Streamlit**, **scikit-learn**, and the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db).

---

## Live Demo
[Click here to try the app](https://amrutha2912-spotify-hybrid-music-recommender.streamlit.app)

---

## Project Overview

This project combines two popular recommendation techniques:

1. **Content-Based Filtering**
   - Uses audio features such as *danceability, energy, valence, tempo,* etc.
   - Finds similar tracks using cosine similarity via `NearestNeighbors`.

2. **Collaborative Filtering**
   - Simulates user–song rating data.
   - Learns hidden user and song representations using **Non-Negative Matrix Factorization (NMF)**.

3. **Hybrid Recommendation**
   - Merges both approaches using a tunable weight `α`:
     ```
     final_score = α * content_score + (1 - α) * collaborative_score
     ```

---

## Features

✅ Enter any song title and get top-K similar recommendations  
✅ Adjust the weight between **content** and **collaborative** influence  
✅ Explore simulated user profiles  
✅ Fast, memory-efficient model (no giant similarity matrix)  
✅ Built-in UI with Streamlit for easy exploration  


