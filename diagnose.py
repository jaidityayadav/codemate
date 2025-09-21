#!/usr/bin/env python3
"""
Diagnostic script to identify what's causing the memory issues
"""

import psutil
import os
import sys

def check_system_resources():
    """Check available system resources"""
    print("=== SYSTEM RESOURCES ===")
    
    # Memory info
    memory = psutil.virtual_memory()
    print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
    print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
    print(f"Used RAM: {memory.used / (1024**3):.1f} GB")
    print(f"Memory usage: {memory.percent}%")
    
    # Disk space
    disk = psutil.disk_usage('/')
    print(f"Disk free: {disk.free / (1024**3):.1f} GB")
    
    # CPU info
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"CPU usage: {psutil.cpu_percent(interval=1)}%")
    
    print()

def test_model_loading():
    """Test loading different models to see where the issue occurs"""
    print("=== TESTING MODEL LOADING ===")
    
    try:
        print("1. Testing basic imports...")
        import torch
        print("✅ Torch imported successfully")
        
        from transformers import AutoTokenizer, AutoModel
        print("✅ Transformers imported successfully")
        
        print("2. Testing tokenizer loading...")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Tokenizer loaded successfully")
        
        print("3. Testing model loading...")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Model loaded successfully")
        
        print("4. Testing basic inference...")
        inputs = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Basic inference successful")
        
        print("5. Testing with your data...")
        with open("data/a.txt", "r") as f:
            content = f.read()
        
        # Test chunking
        from utils.chunker import chunk_text
        chunks = chunk_text(content, chunk_size=400, overlap=50)
        print(f"✅ Created {len(chunks)} chunks from your file")
        
        # Test encoding one chunk
        test_chunk = chunks[0][:400]  # Take first chunk, limit size
        inputs = tokenizer(test_chunk, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        print("✅ Successfully encoded one chunk")
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()

def check_docker():
    """Check Docker containers"""
    print("=== DOCKER STATUS ===")
    os.system("docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'")
    print()

if __name__ == "__main__":
    check_system_resources()
    check_docker()
    test_model_loading()