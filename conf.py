import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

