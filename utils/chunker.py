import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Splits text into overlapping chunks.
    chunk_size = number of characters per chunk
    overlap = number of overlapping characters between chunks
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")
    
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    
    chunks = []
    start = 0
    max_iterations = len(text) // (chunk_size - overlap) + 10  # Safety limit
    iterations = 0
    
    while start < len(text) and iterations < max_iterations:
        iterations += 1
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        
        # Calculate next start position
        next_start = end - overlap
        
        # Ensure we always make progress
        if next_start <= start:
            next_start = start + 1
        
        start = next_start
    
    return chunks

if __name__ == "__main__":
    sample = "This is a long text that should be split into smaller overlapping chunks for embeddings."
    print(chunk_text(sample, chunk_size=20, overlap=5))
