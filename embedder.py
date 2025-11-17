from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from web_Scrapper import get_chennai_news

# ======================
# Step 1: Get Data
# ======================
df = get_chennai_news()
print("STEP 1: Data loaded")
print(df.head())

# ======================
# Step 2: Combine columns
# ======================
print("\nSTEP 2: Combining 'title' and 'news' columns...")
texts = (df["title"].fillna("") + "\n\n" + df["news"].fillna("")).tolist()
print(f"Total rows to embed: {len(texts)}")

# ======================
# Step 3: Build Embeddings
# ======================
print("\nSTEP 3: Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded. Creating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
)
vector_size = embeddings.shape[1]
print(f"Embeddings created. Vector size = {vector_size}")

# ==============================
# Step 4: Connect to Qdrant
# ==============================
print("\nSTEP 4: Connecting to local Qdrant (Docker)...")
client = QdrantClient(
    url="http://localhost:6333",
    api_key=None,
)
print("Connected to Qdrant.")

collection_name = "news_hnsw"

# =========================================
# Step 5: Create / Recreate collection
# =========================================
print(f"\nSTEP 5: Ensuring collection '{collection_name}' exists...")

if client.collection_exists(collection_name=collection_name):
    print(f"Collection '{collection_name}' already exists. Deleting it...")
    client.delete_collection(collection_name=collection_name)

print(f"Creating collection '{collection_name}'...")
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=vector_size,
        distance=models.Distance.COSINE,  # HNSW under the hood
    ),
)

print(f"Collection '{collection_name}' is ready.")

# ======================
# Step 6: Prepare Points
# ======================
print("\nSTEP 6: Preparing points for upsert...")
points = []

for idx, (row, vector) in enumerate(zip(df.itertuples(index=False), embeddings)):
    points.append(
        models.PointStruct(
            id=int(idx),
            vector=vector.tolist(),
            payload={
                "title": row.title,
                "news": row.news,
            },
        )
    )

print(f"Prepared {len(points)} points.")

# ======================
# Step 7: Upsert
# ======================
print("\nSTEP 7: Upserting points into Qdrant...")
client.upsert(
    collection_name=collection_name,
    wait=True,
    points=points,
)
print("Upsert completed ✅")

# ======================
# Step 8: Sanity-check search
# ======================
print("\nSTEP 8: Running a test search...")

query_text = "metabolic wellness center at hospital"
query_vector = model.encode([query_text])[0].tolist()

search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,      # <- correct argument name
    limit=3,
    with_payload=True,
)

print("\nSearch results:")
for r in search_results.points:
    print(f"- Score: {r.score:.4f}")
    print(f"  Title: {r.payload.get('title')}")
    print()
