import os

QDRANT_HOST = os.getenv("QDRANT_HOST")
QDRANT_PORT = os.getenv("QDRANT_PORT")

DATABASE_URL = os.getenv("DATABASE_URL")

REDIS_CACHE_HOST = os.getenv("REDIS_CACHE_HOST")
REDIS_CACHE_PORT = os.getenv("REDIS_CACHE_PORT")
