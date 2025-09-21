#!/usr/bin/env python3
"""
Minimal test to isolate the memory leak issue
"""

import psutil
import gc
import os

def monitor_memory():
    """Monitor current memory usage"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def test_chunking():
    """Test if chunking is causing issues"""
    print("=== TESTING CHUNKING ===")
    print(f"Initial memory: {monitor_memory():.1f} MB")
    
    # Read the file
    with open("data/a.txt", "r") as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    print(f"After reading file: {monitor_memory():.1f} MB")
    
    # Test chunking
    from utils.chunker import chunk_text
    chunks = chunk_text(content, chunk_size=400, overlap=50)
    
    print(f"Number of chunks: {len(chunks)}")
    print(f"After chunking: {monitor_memory():.1f} MB")
    
    # Check chunk sizes
    chunk_sizes = [len(chunk) for chunk in chunks]
    print(f"Chunk sizes: {chunk_sizes[:5]}...")  # Show first 5
    
    return chunks

def test_tokenizer_only():
    """Test just the tokenizer without model"""
    print("\n=== TESTING TOKENIZER ONLY ===")
    print(f"Initial memory: {monitor_memory():.1f} MB")
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"After loading tokenizer: {monitor_memory():.1f} MB")
    
    # Test with a small chunk
    test_text = "This is a test sentence."
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=256)
    
    print(f"After tokenizing: {monitor_memory():.1f} MB")
    print(f"Input shape: {inputs['input_ids'].shape}")
    
    return tokenizer

def test_model_loading():
    """Test model loading separately"""
    print("\n=== TESTING MODEL LOADING ===")
    print(f"Initial memory: {monitor_memory():.1f} MB")
    
    from transformers import AutoModel
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"After loading model: {monitor_memory():.1f} MB")
    
    return model

def test_embedding_generation():
    """Test the actual embedding generation that's likely causing issues"""
    print("\n=== TESTING EMBEDDING GENERATION ===")
    
    chunks = test_chunking()
    tokenizer = test_tokenizer_only()
    
    print(f"Before model loading: {monitor_memory():.1f} MB")
    model = test_model_loading()
    
    print(f"Testing with first chunk: '{chunks[0][:50]}...'")
    
    import torch
    with torch.no_grad():
        inputs = tokenizer(chunks[0], return_tensors="pt", truncation=True, max_length=256)
        print(f"After tokenizing chunk: {monitor_memory():.1f} MB")
        
        outputs = model(**inputs)
        print(f"After model forward: {monitor_memory():.1f} MB")
        
        # Extract embeddings (this might be where the leak is)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        
        print(f"Token embeddings shape: {token_embeddings.shape}")
        print(f"After extracting token embeddings: {monitor_memory():.1f} MB")
        
        # Mean pooling - this operation might be causing memory explosion
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        print(f"After mask expansion: {monitor_memory():.1f} MB")
        
        embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        print(f"After mean pooling: {monitor_memory():.1f} MB")
        
        # Normalization
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        print(f"After normalization: {monitor_memory():.1f} MB")
        
        # Convert to numpy
        result = embeddings.cpu().numpy()
        print(f"After numpy conversion: {monitor_memory():.1f} MB")
        print(f"Final embedding shape: {result.shape}")

if __name__ == "__main__":
    try:
        test_embedding_generation()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()