from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
import numpy as np
import uuid
import os
from dotenv import load_dotenv

load_dotenv()

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

COLLECTION_NAME = "documents"

def init_qdrant():
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )

def upload_vectors(user_id: str, document_id: str, vectors: list, metadata: list):
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "user_id": user_id,  # 👈 добавили user_id
                "document_id": document_id,
                **meta
            }
        )
        for vec, meta in zip(vectors, metadata)
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Загружаем {len(vectors)} векторов от user_id: {user_id}, document_id: {document_id}")



def search_vectors(query_vector, top_k=5):
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )
    return hits

def delete_vectors_by_document_id(document_id: str, user_id: str = None):
    conditions = [
        FieldCondition(
            key="document_id",
            match=MatchValue(value=document_id)
        )
    ]
    if user_id:
        conditions.append(
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id)
            )
        )

    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(must=conditions)
    )


