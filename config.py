from pymilvus import connections

# Connect to Milvus (local docker: localhost:19530)
connections.connect("default", host="127.0.0.1", port="19530")

# Lightweight Hugging Face embedding model (correct model name)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # This works with transformers
COLLECTION_NAME = "research_docs"
