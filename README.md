# Document Search and Retrieval System

A semantic search system using Milvus vector database and Ollama embeddings for intelligent document retrieval.

## Quick Start Guide

### Prerequisites
- Docker installed
- Python 3.8+
- Ollama installed

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Milvus Database

Start the Milvus vector database using Docker:

```bash
# Start Milvus containers (etcd, minio, milvus)
docker start milvus-etcd milvus-minio milvus-standalone
```

If containers don't exist, create them first:

```bash
# Pull and run Milvus with dependencies
docker run -d --name milvus-etcd quay.io/coreos/etcd:v3.5.5 etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd

docker run -d --name milvus-minio -p 9001:9001 -p 9000:9000 minio/minio:RELEASE.2023-03-20T20-16-18Z server /minio_data --console-address ":9001"

docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 --link milvus-etcd:etcd --link milvus-minio:minio milvusdb/milvus:v2.3.0 milvus run standalone
```

**Verify Milvus is running:**
```bash
docker ps | grep milvus
```

### 3. Start Ollama and Install Embedding Model

```bash
# Start Ollama service
ollama serve
```

In a new terminal, pull the Nomic embedding model:
```bash
ollama pull nomic-embed-text
```

**Verify Ollama is working:**
```bash
ollama list
```

### 4. Prepare Your Documents

Place your text documents (`.txt`, `.md`) in the `data/` directory:

```bash
mkdir -p data/
# Copy your documents to data/
cp your-documents.txt data/
```

### 5. Ingest Documents (REQUIRED FIRST STEP)

Run the ingestion script to process and store document embeddings:

```bash
python3 ingest.py
```

**Expected output:**
```
Starting document ingestion...
Using Ollama model: nomic-embed-text
Connected to Ollama successfully!
Dropping existing collection 'research_docs'...
Created collection: research_docs
Found X files to process
Processing: your-document.txt
   Split into X chunks
   Inserted batch 1/X
All documents ingested successfully!
```

### 6. Run the Application

After successful ingestion, you can query your documents:

#### Option A: Interactive Research Agent
```bash
python3 main.py
```

This starts an interactive session where you can ask questions:
```
Deep Researcher Agent Ready

Enter your query (or 'exit'): What are the main topics?
```

#### Option B: Single Query Search
```bash
python3 search.py "your search query here"
```

#### Option C: Test Multiple Queries
```bash
python3 test_query.py
```

## Troubleshooting

### Milvus Issues
```bash
# Check if containers are running
docker ps | grep milvus

# Restart Milvus if needed
docker restart milvus-etcd milvus-minio milvus-standalone

# Check logs
docker logs milvus-standalone
```

### Ollama Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve

# Re-pull the model if needed
ollama pull nomic-embed-text
```

### Memory Issues
If you encounter memory problems during ingestion:
- Reduce `BATCH_SIZE` in `ingest.py`
- Process smaller documents
- Restart Docker containers

### Connection Issues
```bash
# Test Milvus connection
python3 -c "from pymilvus import connections; connections.connect('default', host='localhost', port='19530'); print('Connected!')"

# Test Ollama connection
curl http://localhost:11434/api/tags
```

## Project Structure

```
assignment3/
├── data/              # Place your documents here
├── ingest.py          # Document ingestion script (run first)
├── main.py            # Interactive research agent
├── search.py          # Single query search
├── test_query.py      # Test multiple queries
├── config.py          # Configuration settings
├── retriever.py       # Vector search logic
├── agent.py           # Research agent logic
├── utils/
│   └── chunker.py     # Text chunking utilities
└── requirements.txt   # Python dependencies
```

## Workflow Summary

1. **Setup**: Install Docker, Python deps, Ollama
2. **Start Services**: Milvus (Docker) + Ollama + nomic-embed-text
3. **Add Documents**: Place files in `data/` directory  
4. **Ingest**: Run `python3 ingest.py` (REQUIRED FIRST)
5. **Query**: Run `python3 main.py` for interactive search

## Usage Examples

```bash
# Example queries you can try:
python3 search.py "What are the main concepts?"
python3 search.py "Tell me about the challenges"
python3 search.py "Summarize the key points"
```