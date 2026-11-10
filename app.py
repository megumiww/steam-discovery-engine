import streamlit as st

MOCK_DATA = [
    {"name": "Starfall Frontier", "tags": "Space, Exploration, Sci-Fi", "score": 0.95},
    {"name": "Ironwood Survival", "tags": "Survival, Crafting, Open World", "score": 0.90},
    {"name": "Mystic Realms", "tags": "RPG, Fantasy, Magic", "score": 0.86},
    {"name": "Neon Racer", "tags": "Racing, Futuristic, Arcade", "score": 0.82},
    {"name": "Farm Tales", "tags": "Farming, Simulation, Relaxing", "score": 0.78},
]

MODEL_MAP = {
    "Popularity": "PopularityBaseline",
    "TF-IDF": "TFIDFRetriever",
    "BM25": "BM25Retriever",
    "Hybrid": "HybridRanker",
}


def _load_model(model_name):
    try:
        if model_name == "Popularity":
            from src.steam_discovery.popularity_baseline import PopularityBaseline

            return PopularityBaseline()
        if model_name == "TF-IDF":
            from src.steam_discovery.tfidf_retriever import TFIDFRetriever

            return TFIDFRetriever()
        if model_name == "BM25":
            from src.steam_discovery.bm25_retriever import BM25Retriever

            return BM25Retriever()
        if model_name == "Hybrid":
            from src.steam_discovery.hybrid_ranker import HybridRanker

            return HybridRanker()
    except Exception:
        return None

    return None


def _call_model(model, query, top_k):
    if model is None:
        return []

    methods = ("recommend", "retrieve", "search", "get_recommendations", "rank")

    for method_name in methods:
        method = getattr(model, method_name, None)
        if not callable(method):
            continue

        for call in (
            lambda: method(query, top_k),
            lambda: method(query=query, top_k=top_k),
            lambda: method(query, k=top_k),
            lambda: method(query),
        ):
            try:
                result = call()
                if result is not None:
                    return result
            except TypeError:
                continue
            except Exception:
                return []

    return []


def _extract_value(item, keys, default=None):
    if isinstance(item, dict):
        for key in keys:
            if key in item:
                return item[key]

    for key in keys:
        if hasattr(item, key):
            return getattr(item, key)

    return default


def _format_tags(value):
    if value is None:
        return "未知"
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(tag) for tag in value)
    return str(value)


def _normalize_results(results, top_k):
    if results is None:
        return []

    if isinstance(results, dict):
        for key in ("results", "recommendations", "items", "games", "data"):
            if key in results:
                results = results[key]
                break
        else:
            results = [results]

    if not isinstance(results, (list, tuple)):
        try:
            results = list(results)
        except Exception:
            results = [results]

    normalized = []
    for item in list(results)[:top_k]:
        if isinstance(item, (list, tuple)):
            name = item[0] if len(item) > 0 else "未知游戏"
            score = item[1] if len(item) > 1 else 0.0
            tags = item[2] if len(item) > 2 else "未知"
        else:
            name = _extract_value(item, ("name", "game_name", "title", "app_name", "game"), "未知游戏")
            tags = _extract_value(item, ("tags", "genres", "genre", "categories"), "未知")
            score = _extract_value(item, ("score", "similarity", "rank_score", "relevance", "rating"), 0.0)

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.0

        normalized.append({"name": str(name), "tags": _format_tags(tags), "score": score})

    return normalized


def main():
    st.set_page_config(page_title="Steam Discovery Engine", page_icon="🎮")
    st.title("🎮 Steam Discovery Engine")
    st.sidebar.header("推荐设置")

    model_name = st.sidebar.selectbox(
        "选择推荐模型",
        options=list(MODEL_MAP.keys()),
        format_func=lambda name: f"{name} ({MODEL_MAP[name]})",
    )
    top_k = st.sidebar.slider("推荐数量", min_value=1, max_value=10, value=5)
    query = st.text_input("输入游戏关键词", placeholder="例如：survival crafting")

    if st.button("推荐", type="primary"):
        if not query.strip():
            st.warning("请输入游戏关键词。")
            return

        model = _load_model(model_name)
        results = _normalize_results(_call_model(model, query.strip(), top_k), top_k)

        if not results:
            results = MOCK_DATA[:top_k]
            st.info("真实检索结果不可用，已使用内置 Mock 数据。")

        st.subheader(f"推荐结果 · {model_name}")
        for index, result in enumerate(results, start=1):
            with st.container():
                col1, col2, col3 = st.columns([3, 4, 1])
                col1.write(f"**{index}. {result['name']}**")
                col2.write(f"标签：{result['tags']}")
                col3.write(f"{result['score']:.3f}")
                st.divider()


if __name__ == "__main__":
    main()
