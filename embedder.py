from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models

from conf import MODEL_NAME, QDRANT_URL, COLLECTION_NAME
from web_Scrapper import get_chennai_news
import traceback

# Load model & client ONCE at module import
try:
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")

    print("Connecting to Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=None)
    print("Connected to Qdrant.")
except Exception as e:
    print("Initialization error:", e)
    model = None
    client = None


def _get_existing_titles_and_max_id():
    """Scroll the collection and return (set_of_titles, max_existing_id)."""
    existing_titles = set()
    max_id = 0

    try:
        next_offset = None
        while True:
            points, next_offset = client.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=next_offset,
            )

            for p in points:
                # collect titles
                title = (p.payload or {}).get("title")
                if title:
                    existing_titles.add(title)

                # track max id for integer ids
                try:
                    pid = int(p.id)
                    if pid > max_id:
                        max_id = pid
                except (ValueError, TypeError):
                    # if existing ids are not ints, ignore for max_id
                    pass

            if next_offset is None:
                break
    except Exception as e:
        print("Error while reading existing titles from Qdrant:", e)

    return existing_titles, max_id


def index_chennai_news():
    """
    Scrape → embed → store in Qdrant (append-only) → run a test search.
    - Does NOT delete collection
    - Skips documents whose title already exists in Qdrant
    - Also deduplicates titles inside the current batch

    Returns (response_dict, status_code)
    """
    try:
        if model is None or client is None:
            return {
                "status": "error",
                "message": "Model or Qdrant client not initialized",
            }, 500

        # Step 1: Get news
        df = get_chennai_news()
        if df is None or df.empty:
            return {"status": "error", "message": "No news data scraped"}, 500

        # Drop duplicates inside this batch based on title
        df = df.dropna(subset=["title"])
        df = df.drop_duplicates(subset=["title"], keep="first")

        # Build text list
        texts = (df["title"].fillna("") + "\n\n" + df["news"].fillna("")).tolist()
        if not texts:
            return {"status": "error", "message": "No text to embed"}, 400

        # Step 2: Embeddings
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vector_size = embeddings.shape[1]

        # Step 3: Ensure collection exists (no delete)
        if not client.collection_exists(collection_name=COLLECTION_NAME):
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

        # Step 4: Load existing titles + max id from Qdrant
        existing_titles, max_existing_id = _get_existing_titles_and_max_id()
        existing_count_before = len(existing_titles)
        print(f"Existing titles in DB: {existing_count_before}, max id: {max_existing_id}")

        # Step 5: Prepare only NEW points
        points = []
        skipped_existing = 0
        seen_titles_this_run = set()
        next_id = max_existing_id + 1

        for (row, vector) in zip(df.itertuples(index=False), embeddings):
            title = row.title

            # Skip if already in DB
            if title in existing_titles:
                skipped_existing += 1
                continue

            # Skip duplicates within this run (extra safety)
            if title in seen_titles_this_run:
                skipped_existing += 1
                continue

            seen_titles_this_run.add(title)

            points.append(
                models.PointStruct(
                    id=next_id,               # ✅ valid integer id
                    vector=vector.tolist(),
                    payload={
                        "title": row.title,
                        "news": row.news,
                    },
                )
            )
            next_id += 1

        if not points:
            return {
                "status": "success",
                "message": "No new articles to index. All titles already exist.",
                "collection": COLLECTION_NAME,
                "total_existing_before": existing_count_before,
                "new_indexed": 0,
                "skipped_existing_titles": skipped_existing,
            }, 200

        # Step 6: Upsert only new points
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points,
        )

        # Step 7: Test search
        test_query = "metabolic wellness hospital"
        q_vec = model.encode([test_query])[0].tolist()

        search = client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=3,
            with_payload=True,
        )

        test_hits = [
            {
                "id": p.id,
                "score": p.score,
                "title": (p.payload or {}).get("title"),
            }
            for p in search.points
        ]

        return {
            "status": "success",
            "message": "News indexed into Qdrant successfully",
            "collection": COLLECTION_NAME,
            "total_existing_before": existing_count_before,
            "new_indexed": len(points),
            "skipped_existing_titles": skipped_existing,
            "vector_size": vector_size,
            "example_query": test_query,
            "example_hits": test_hits,
        }, 201

    except Exception as e:
        print("Error indexing news:")
        traceback.print_exc()
        return {
            "status": "error",
            "message": str(e),
        }, 500

