# Steam Discovery Engine

An algorithm-focused Python project for searching and recommending Steam games.

The project will evolve in layers:

1. Popularity baseline.
2. TF-IDF and BM25 content retrieval.
3. Collaborative filtering and matrix factorization.
4. Hybrid ranking with offline evaluation.

## Step 1: Data Contract and Popularity Baseline

The first version uses two CSV files:

- `data/raw/games.csv` — game metadata and rating statistics.
- `data/raw/interactions.csv` — user-game feedback used to exclude already-seen games.

Run the baseline after installing the local package:

```powershell
python -m pip install -e . --no-deps
python -m steam_discovery.cli --games data/raw/games.csv --interactions data/raw/interactions.csv --user u1 --top-k 5
```

The baseline ranks games with a Bayesian rating estimate and a small popularity signal. It is intentionally simple: later algorithms must beat this baseline in offline evaluation.

## Step 2: TF-IDF Content Retrieval

The second model builds TF-IDF vectors from each game's title, genres, tags, and description. It supports both text search and profile-based recommendations using cosine similarity.

```powershell
python -m steam_discovery.cli --games data/raw/games.csv --interactions data/raw/interactions.csv --user u1 --mode content --top-k 5
python -m steam_discovery.cli --games data/raw/games.csv --interactions data/raw/interactions.csv --user u1 --mode content --query 'co-op survival' --top-k 5
```

## Step 3: BM25 Search Ranking

The third model builds an inverted index and ranks lexical matches with BM25. Unlike TF-IDF cosine search, BM25 normalizes term frequency and document length, making it a stronger baseline for keyword search.

```powershell
python -m steam_discovery.cli --games data/raw/games.csv --interactions data/raw/interactions.csv --user u1 --mode bm25 --query 'co-op survival' --top-k 5
```

## Step 4: Hybrid Ranking and Evaluation

The hybrid ranker combines normalized BM25, TF-IDF, and popularity signals. The evaluation module provides Precision@K, Recall@K, Hit Rate, and NDCG@K for offline experiments.

```powershell
python -m steam_discovery.cli --games data/raw/games.csv --interactions data/raw/interactions.csv --user u1 --mode hybrid --query 'co-op survival' --top-k 5
```
