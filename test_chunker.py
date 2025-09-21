#!/usr/bin/env python3

def test_chunker_bug():
    """Test the chunker with problematic parameters"""
    
    # This will cause infinite loop: overlap (50) > chunk_size (400) is false, but let's test edge cases
    
    text = "A" * 1000  # Simple text
    chunk_size = 400
    overlap = 50
    
    print(f"Text length: {len(text)}")
    print(f"Chunk size: {chunk_size}, Overlap: {overlap}")
    
    chunks = []
    start = 0
    iterations = 0
    
    while start < len(text) and iterations < 100:  # Safety limit
        iterations += 1
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        
        old_start = start
        start = end - overlap
        
        print(f"Iteration {iterations}: start={old_start} -> end={end} -> new_start={start}")
        
        if start < 0:
            start = 0
        
        # Check for infinite loop condition
        if start <= old_start and len(text) > chunk_size:
            print(f"❌ INFINITE LOOP DETECTED! start={start}, old_start={old_start}")
            break
    
    print(f"Created {len(chunks)} chunks in {iterations} iterations")

if __name__ == "__main__":
    test_chunker_bug()