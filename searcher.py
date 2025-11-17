from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer




def search_news(query: str, top_k: int = 5):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"\nSearching for: {query!r}")

    client = QdrantClient(
        url="http://localhost:6333",
        api_key=None,
    )
    # 1) Embed the query using the SAME model
    query_vector = model.encode([query])[0].tolist()

    # 2) Search in Qdrant
    result = client.query_points(
        collection_name="news_hnsw",
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,  # set True if you want to inspect vectors too
    )

    print(f"Found {len(result.points)} results:\n")
    for idx, point in enumerate(result.points, start=1):
        title = point.payload.get("title", "")[:120]
        news_snippet = point.payload.get("news", "")[:200].replace("\n", " ")
        print(f"{idx}. (score={point.score:.4f}) {title}")
        print(f"   {news_snippet}")
        print()

    return result



print(search_news(query="cricket"))