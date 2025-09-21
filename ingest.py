import os
import gc
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility
from config import COLLECTION_NAME, EMBED_MODEL
from utils.chunker import chunk_text

BATCH_SIZE = 4  # Very small batch size to prevent memory issues

class MemoryEfficientEmbedder:
    def __init__(self, model_name):
        print(f"🔄 Loading embedding model: {model_name}")
        self.device = torch.device('cpu')  # Force CPU to avoid GPU memory issues
        print(f"📱 Using device: {self.device}")
        
        # Load model components
        print("📥 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("📥 Loading model...")
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ Model loaded successfully!")
        
    def encode_single(self, text):
        """Encode a single text to prevent memory overload"""
        with torch.no_grad():
            # Tokenize with limited length
            inputs = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=256,  # Reduced max length
                return_tensors='pt'
            ).to(self.device)
            
            # Get embeddings
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Clear memory immediately
            del inputs, outputs, token_embeddings, input_mask_expanded
            
            return embeddings.cpu().numpy()[0]
    
    def encode(self, texts):
        """Encode texts one by one to prevent memory issues"""
        embeddings = []
        for i, text in enumerate(texts):
            if i % 2 == 0:  # Progress every 2 items
                print(f"    🔄 Processing {i+1}/{len(texts)}")
            
            embedding = self.encode_single(text)
            embeddings.append(embedding)
            
            # Force garbage collection every item
            gc.collect()
            
        return np.array(embeddings)

def create_collection():
    # Drop existing collection to avoid conflicts
    if utility.has_collection(COLLECTION_NAME):
        print(f"🗑️ Dropping existing collection '{COLLECTION_NAME}'...")
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
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
        print("🚀 Starting document ingestion...")
        embedder = MemoryEfficientEmbedder(EMBED_MODEL)
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

                chunks = chunk_text(raw_text, chunk_size=400, overlap=50)  # Smaller chunks
                print(f"📝 Split into {len(chunks)} chunks")

                # Process in very small batches
                for i in range(0, len(chunks), BATCH_SIZE):
                    try:
                        batch = chunks[i:i+BATCH_SIZE]
                        # Truncate chunks for VARCHAR limit
                        batch = [chunk[:800] for chunk in batch]  # Reduced limit
                        
                        print(f"  🔄 Processing batch {i//BATCH_SIZE+1}/{(len(chunks)-1)//BATCH_SIZE+1}")
                        embeddings = embedder.encode(batch)
                        embeddings_list = embeddings.tolist()
                        
                        collection.insert([embeddings_list, batch])
                        
                        print(f"  ✅ Inserted batch {i//BATCH_SIZE+1}")
                        
                        # Aggressive garbage collection
                        gc.collect()
                        
                    except Exception as e:
                        print(f"  ❌ Error processing batch {i//BATCH_SIZE+1}: {e}")
                        continue

            except Exception as e:
                print(f"❌ Error processing file {fname}: {e}")
                continue

        collection.flush()
        print("\n🎉 All documents ingested successfully!")
        
        # Final memory cleanup
        del embedder
        gc.collect()
        
    except Exception as e:
        print(f"❌ Fatal error during ingestion: {e}")
        raise

if __name__ == "__main__":
    ingest_docs()
