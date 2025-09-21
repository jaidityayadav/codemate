import os
import gc
import requests
import json
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from config import COLLECTION_NAME, EMBED_MODEL
from utils.chunker import chunk_text

BATCH_SIZE = 8  # Process multiple chunks at once

class OllamaEmbedder:
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        print(f"� Using Ollama model: {model_name}")
        
        # Test connection
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print("✅ Connected to Ollama successfully!")
            else:
                raise Exception("Cannot connect to Ollama")
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            print("Make sure Ollama is running: ollama serve")
            raise
    
    def encode(self, texts):
        """Encode texts using Ollama API"""
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                # Prepare the request
                payload = {
                    "model": self.model_name,
                    "prompt": text
                }
                
                # Make request to Ollama
                response = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    embedding = result["embedding"]
                    embeddings.append(embedding)
                    
                    if (i + 1) % 5 == 0:  # Progress every 5 items
                        print(f"    🔄 Processed {i+1}/{len(texts)} embeddings")
                else:
                    print(f"❌ Error getting embedding for text {i+1}: {response.text}")
                    # Use zero vector as fallback
                    embeddings.append([0.0] * 768)  # Nomic-embed-text is 768 dimensions
                    
            except Exception as e:
                print(f"❌ Error processing text {i+1}: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)
        
        return embeddings

def create_collection():
    # Drop existing collection to avoid conflicts
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️ Dropping existing collection '{COLLECTION_NAME}'...")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Nomic-embed-text is 768 dimensions
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
    ]
    schema = CollectionSchema(fields, description="Research documents")
    collection = Collection(COLLECTION_NAME, schema)
    
    # Create index for better search performance
    index_params = {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT", 
        "params": {"nlist": 64}  # Smaller nlist for small datasets
    }
    collection.create_index("embedding", index_params)
    collection.load()
    print(f"✅ Created collection: {COLLECTION_NAME}")
    return collection

def ingest_docs():
    try:
        print("🚀 Starting document ingestion with Ollama...")
        embedder = OllamaEmbedder()
        collection = create_collection()

        data_dir = "data"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory '{data_dir}' not found!")
            return

        files = [f for f in os.listdir(data_dir) if f.endswith(('.txt', '.md'))]
        if not files:
            print(f"❌ No text files found in '{data_dir}'")
            return

        print(f"📚 Found {len(files)} files to process")

        for fname in files:
            try:
                path = os.path.join(data_dir, fname)
                print(f"\n📄 Processing: {fname}")
                
                with open(path, "r", encoding='utf-8') as f:
                    raw_text = f.read()

                if len(raw_text.strip()) == 0:
                    print(f"⚠️ Skipping empty file: {fname}")
                    continue

                chunks = chunk_text(raw_text, chunk_size=400, overlap=50)
                print(f"📝 Split into {len(chunks)} chunks")

                # Process in batches
                for i in range(0, len(chunks), BATCH_SIZE):
                    try:
                        batch = chunks[i:i+BATCH_SIZE]
                        # Truncate chunks for VARCHAR limit
                        batch = [chunk[:900] for chunk in batch]  # Leave some buffer
                        
                        print(f"  🔄 Processing batch {i//BATCH_SIZE+1}/{(len(chunks)-1)//BATCH_SIZE+1}")
                        embeddings = embedder.encode(batch)
                        
                        collection.insert([embeddings, batch])
                        
                        print(f"  ✅ Inserted batch {i//BATCH_SIZE+1}")
                        
                        # Light garbage collection
                        gc.collect()
                        
                    except Exception as e:
                        print(f"  ❌ Error processing batch {i//BATCH_SIZE+1}: {e}")
                        continue

            except Exception as e:
                print(f"❌ Error processing file {fname}: {e}")
                continue

        collection.flush()
        print("\n🎉 All documents ingested successfully!")
        
    except Exception as e:
        print(f"❌ Fatal error during ingestion: {e}")
        raise

if __name__ == "__main__":
    ingest_docs()
