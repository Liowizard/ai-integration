import traceback

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from conf import MODEL_NAME, QDRANT_URL, COLLECTION_NAME

try:
    print("Loading embedding model for search...")
    model = SentenceTransformer(MODEL_NAME)
    print("Search model loaded.")

    print("Connecting to Qdrant for search...")
    client = QdrantClient(url=QDRANT_URL, api_key=None)
    print("Search Qdrant client ready.")
except Exception as e:
    print("Search initialization error:", e)
    model = None
    client = None


def search_news(query: str, top_k: int = 5):
    """
    Semantic search in Qdrant for news articles.
    Returns (response_dict, http_status_code)
    """
    try:
        if not query or not query.strip():
            return {
                "status": "error",
                "message": "Query string cannot be empty.",
            }, 400

        if model is None or client is None:
            return {
                "status": "error",
                "message": "Model or Qdrant client not initialized.",
            }, 500

        query = query.strip()
        print(f"\nSearching for: {query!r} (top_k={top_k})")

        # 1) Embed the query using the SAME model
        query_vector = model.encode([query])[0].tolist()

        # 2) Search in Qdrant
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,  # set True if you want to inspect vectors
        )

        hits = []
        for idx, point in enumerate(result.points, start=1):
            payload = point.payload or {}
            title = payload.get("title", "") or ""
            news = payload.get("news", "") or ""

            hits.append(
                {
                    "rank": idx,
                    "id": point.id,
                    "score": point.score,
                    "title": title,
                    "news_snippet": news.replace("\n", " ")[:300],
                }
            )

        print(f"Found {len(hits)} results.\n")

        return {
            "status": "success",
            "query": query,
            "results_count": len(hits),
            "results": hits,
        }, 200

    except Exception as e:
        print("Error in search_news:")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
        }, 500
