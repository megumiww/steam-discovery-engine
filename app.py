from pathlib import Path

import streamlit as st


MOCK_DATA = [
    {"title": "Starfall Frontier", "tags": "co-op,sci-fi,survival", "score": 0.92},
    {"title": "Ironwood Survival", "tags": "survival,open-world", "score": 0.87},
    {"title": "Mystic Realms", "tags": "rpg,fantasy", "score": 0.81},
    {"title": "Neon Racer", "tags": "racing,cyberpunk", "score": 0.76},
    {"title": "Farm Tales", "tags": "farming,simulation", "score": 0.71},
]

MODEL_NAMES = {
    "Popularity": "PopularityBaseline",
    "TF-IDF": "TfidfRetriever",
    "BM25": "BM25Retriever",
    "Hybrid": "HybridRanker",
}


def _load_data():
    from src.steam_discovery.data import load_games, load_interactions

    data_dir = Path(__file__).resolve().parent / "data" / "raw"
    games = load_games(data_dir / "games.csv")
    interactions = load_interactions(data_dir / "interactions.csv")
    return games, interactions


def _load_model(model_name):
    try:
        games, interactions = _load_data()

        if model_name == "Popularity":
            from src.steam_discovery.baseline import PopularityBaseline

            return PopularityBaseline(games, interactions)

        if model_name == "TF-IDF":
            from src.steam_discovery.tfidf import TfidfRetriever

            return TfidfRetriever(games, interactions)

        if model_name == "BM25":
            from src.steam_discovery.bm25 import BM25Retriever

            return BM25Retriever(games)

        if model_name == "Hybrid":
            from src.steam_discovery.hybrid import HybridRanker

            return HybridRanker(games, interactions)
    except Exception:
        return None

    return None


def _retrieve(model, model_name, query, top_k):
    if model is None:
        return []

    try:
        if model_name == "Popularity":
            return model.recommend(query, top_k)
        results = model.search(query, top_k)
        if results:
            return results

        # An out-of-vocabulary query should still return real catalog items.
        # Use the project's popularity baseline as a data-backed fallback;
        # Mock data is reserved for import/data failures.
        from src.steam_discovery.baseline import PopularityBaseline

        games, interactions = _load_data()
        return PopularityBaseline(games, interactions).recommend(query, top_k)
    except Exception:
        return []


def _value(item, *keys, default=None):
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                return item[key]

    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)

    return default


def _format_tags(value):
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(tag) for tag in value)
    return str(value or "")


def _display_results(raw_results, top_k):
    displayed = []

    for item in list(raw_results or [])[:top_k]:
        game = _value(item, "game")
        game_id = _value(item, "game_id", "id", "app_id")
        score = _value(item, "score", "similarity", "relevance", default=0.0)

        if game is not None:
            game_id = _value(game, "game_id", "id", "app_id", default=game_id)
            title = _value(game, "title", "name", default=game_id)
            tags = _value(game, "tags", default="")
        else:
            title = _value(item, "title", "name", default=game_id)
            tags = _value(item, "tags", default="")

        if title is None:
            continue

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        displayed.append(
            {
                "title": str(title),
                "tags": _format_tags(tags),
                "score": score,
            }
        )

    return displayed


def main():
    st.set_page_config(page_title="Steam Discovery Engine", page_icon="🎮")
    st.title("🎮 Steam Discovery Engine")
    st.sidebar.header("推荐设置")

    model_name = st.sidebar.selectbox(
        "选择推荐模型",
        list(MODEL_NAMES),
        format_func=lambda name: f"{name} ({MODEL_NAMES[name]})",
    )
    top_k = st.sidebar.slider("推荐数量", min_value=1, max_value=10, value=5)
    query = st.text_input("输入游戏关键词", placeholder="例如：survival crafting")

    if st.button("推荐", type="primary"):
        if not query.strip():
            st.warning("请输入游戏关键词。")
            return

        model = _load_model(model_name)
        results = _display_results(
            _retrieve(model, model_name, query.strip(), top_k),
            top_k,
        )
        using_mock = not results

        if using_mock:
            results = MOCK_DATA[:top_k]

        st.subheader(f"推荐结果 · {model_name}")
        if using_mock:
            st.caption("当前展示模拟数据")

        for index, result in enumerate(results, start=1):
            col1, col2, col3 = st.columns([3, 4, 1])
            col1.write(f"**{index}. {result['title']}**")
            col2.write(f"标签：{result['tags']}")
            col3.write(f"{result['score']:.3f}")
            st.divider()


if __name__ == "__main__":
    main()
