# Steam Discovery Engine

Steam Discovery Engine is a small, reproducible search and recommendation system for Steam games. It combines metadata retrieval with popularity signals and exposes the models through both a command-line interface and a Streamlit app.

## Features

- Popularity baseline with Bayesian rating shrinkage.
- TF-IDF metadata search over titles, genres, tags, and descriptions.
- BM25 lexical search with document-length normalization.
- Hybrid ranking that combines BM25, TF-IDF, and popularity scores.
- Offline evaluation utilities for Precision@K, Recall@K, Hit Rate, and NDCG@K.
- Streamlit UI with model selection, keyword search, Top-K control, and data-backed fallback behavior.

## Project layout

```text
app.py                         Streamlit interface
data/raw/games.csv             Game metadata
data/raw/interactions.csv      User-game interactions
src/steam_discovery/
  baseline.py                  PopularityBaseline
  tfidf.py                     TfidfRetriever
  bm25.py                      BM25Retriever
  hybrid.py                    HybridRanker
  data.py                      CSV-to-dataclass loading
  evaluation.py                Offline evaluation metrics
  cli.py                       Command-line entry point
```

## Setup

The project requires Python 3.11 or newer.

```powershell
python -m pip install -e . --no-deps
```

The Streamlit interface also requires Streamlit in the active environment:

```powershell
python -m pip install streamlit
```

## Run the Streamlit app

From the repository root:

```powershell
python -m streamlit run app.py
```

Enter a query such as `survival crafting`, choose a model, and click **推荐**. The app loads games and interactions through `src/steam_discovery/data.py`, maps result objects back to game titles and tags, and displays the model score. If a query has no lexical match, the app uses the real popularity baseline to fill the result list. Mock data is reserved for import or data-loading failures and is marked in the UI.

## Run the command-line interface

Popularity baseline:

```powershell
python -m steam_discovery.cli `
  --games data/raw/games.csv `
  --interactions data/raw/interactions.csv `
  --user u1 `
  --top-k 5
```

TF-IDF search:

```powershell
python -m steam_discovery.cli `
  --games data/raw/games.csv `
  --interactions data/raw/interactions.csv `
  --mode content `
  --query "co-op survival" `
  --top-k 5
```

BM25 search:

```powershell
python -m steam_discovery.cli `
  --games data/raw/games.csv `
  --interactions data/raw/interactions.csv `
  --mode bm25 `
  --query "co-op survival" `
  --top-k 5
```

Hybrid search:

```powershell
python -m steam_discovery.cli `
  --games data/raw/games.csv `
  --interactions data/raw/interactions.csv `
  --mode hybrid `
  --query "co-op survival" `
  --top-k 5
```

## Data contract

`games.csv` contains `game_id`, `title`, `genres`, `tags`, `description`, `positive_ratings`, `negative_ratings`, and `release_year`. Multi-value genre and tag fields use `|` as the separator.

`interactions.csv` contains `user_id`, `game_id`, `signal`, and `timestamp`. Interaction history is used to exclude seen games for user-based recommendations.

## Architecture

`data.py` converts CSV rows into typed `Game` and `Interaction` records. The retrievers operate on those records and return typed result objects containing the matched game, score, and explanation. `app.py` constructs the selected model with the loaded data and adapts those result objects for display.

## Development

Keep retrieval logic in `src/steam_discovery/`, UI behavior in `app.py`, and evaluation code in `evaluation.py`. Use the same CSV inputs when comparing models so offline metrics remain reproducible.
