from flask import Flask, jsonify, request
from flask_cors import CORS

from conf import DEBUG
from embedder import index_chennai_news
from searcher import search_news

app = Flask(__name__)
CORS(app)

@app.route("/news-sync", methods=["GET"])
def sync_news():
    resp, status = index_chennai_news()
    return jsonify(resp), status



@app.route("/news-search", methods=["GET"])
def searcher():
    # Get query params
    query = request.args.get("query", "").strip()
    top_k = request.args.get("top_k", 5)
    try:
        top_k = int(top_k)
    except:
        top_k = 5

    if not query:
        return jsonify({
            "status": "error",
            "message": "Missing required parameter ?query="
        }), 400
    resp, status = search_news(query=query, top_k=top_k)
    return jsonify(resp), status


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=DEBUG, port=5000)
