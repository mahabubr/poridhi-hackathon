from typing import List
from qdrant_client.models import (
    PointStruct,
    VectorParams,
    Distance,
)
import numpy as np

from .qdrant_client import get_client

qdrant = get_client()

COLLECTION_NAME = "products"


def init_qdrant(vector_size: int):
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def insert_qdrant(embeddings: List[np.ndarray], metadata: List[dict]):
    """Push vectors and metadata into Qdrant."""
    points = [
        PointStruct(id=meta["id_right"], vector=embeddings[i], payload=meta)
        for i, meta in enumerate(metadata)
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)


def search_qdrant(
    query_vector: np.ndarray,
    top_k: int = 5,
):
    """Search Qdrant by vector and optional category filter."""
    query_filter = None

    index = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
    )

    return index
