import numpy as np
import requests
import json
from pymilvus import Collection
from config import COLLECTION_NAME

class OllamaRetriever:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()
        print(f"🔍 Retriever initialized with model: {model_name}")

    def encode_query(self, query):
        """Encode query using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.model_name,
                    "prompt": query
                }
            )
            response.raise_for_status()
            embedding = response.json()["embedding"]
            return embedding
        except Exception as e:
            print(f"❌ Error encoding query: {e}")
            raise

    def search(self, query, top_k=5):
        print(f"🔍 Searching for: '{query}'")
        
        # Encode the query
        query_embedding = self.encode_query(query)
        
        # Search in Milvus
        results = self.collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 16}},
            limit=top_k,
            output_fields=["text"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                formatted_results.append({
                    "text": hit.entity.get("text"),
                    "score": hit.score,
                    "id": hit.id
                })
        
        return formatted_results
        return [hit.entity.get("text") for hit in results[0]]
