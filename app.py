from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from conf import DEBUG
from embedder import index_chennai_news
from searcher import search_news

app = Flask(__name__)
CORS(app)


# ---------- UI Route ----------

@app.route("/", methods=["GET"])
def home():
    """
    Render the main HTML UI.
    """
    return render_template("index.html")


# ---------- API: Sync News ----------

@app.route("/news-sync", methods=["GET"])
def sync_news():
    """
    Trigger scraping + embedding + Qdrant indexing.
    Returns JSON with status, counts, etc.
    """
    resp, status = index_chennai_news()
    return jsonify(resp), status


# ---------- API: Search News ----------

@app.route("/news-search", methods=["GET"])
def news_search():
    """
    Semantic search endpoint.
    Expects ?query=...&top_k=...
    """
    query = request.args.get("query", "").strip()
    top_k = request.args.get("top_k", 5)

    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 5

    if not query:
        return jsonify({
            "status": "error",
            "message": "Missing required parameter ?query=",
        }), 400

    resp, status = search_news(query=query, top_k=top_k)
    return jsonify(resp), status


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=DEBUG, port=5000)
