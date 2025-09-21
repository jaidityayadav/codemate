import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
from config import COLLECTION_NAME, EMBED_MODEL

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()

    def search(self, query, top_k=3):
        qvec = self.model.encode([query]).tolist()
        results = self.collection.search(
            data=qvec,
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["text"]
        )
        return [hit.entity.get("text") for hit in results[0]]
