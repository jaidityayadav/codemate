from pymilvus import connections

# Connect to Milvus (local docker: localhost:19530)
connections.connect("default", host="127.0.0.1", port="19530")

# Ollama embedding model
EMBED_MODEL = "nomic-embed-text"  # Using Ollama Nomic embeddings
COLLECTION_NAME = "research_docs"
