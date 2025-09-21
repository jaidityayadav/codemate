import re

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50):
    """
    Splits text into overlapping chunks.
    chunk_size = number of characters per chunk
    overlap = number of overlapping characters between chunks
    """
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

if __name__ == "__main__":
    sample = "This is a long text that should be split into smaller overlapping chunks for embeddings."
    print(chunk_text(sample, chunk_size=20, overlap=5))
