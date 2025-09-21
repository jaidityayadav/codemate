import re
import heapq
from collections import defaultdict

def summarize(texts, max_sentences=3):
    """
    Summarize multiple documents into top N sentences.
    Uses word frequency scoring.
    """
    joined = " ".join(texts)
    sentences = re.split(r'(?<=[.!?]) +', joined)

    # Build word frequency
    word_freq = defaultdict(int)
    for word in re.findall(r'\w+', joined.lower()):
        word_freq[word] += 1

    # Score sentences
    sent_scores = {}
    for sent in sentences:
        score = 0
        for word in re.findall(r'\w+', sent.lower()):
            score += word_freq.get(word, 0)
        sent_scores[sent] = score

    # Pick top N
    best = heapq.nlargest(max_sentences, sent_scores, key=sent_scores.get)
    return " ".join(best)

if __name__ == "__main__":
    docs = [
        "Milvus is an open-source vector database designed for AI applications.",
        "It can store and search embeddings efficiently.",
        "Researchers use it for similarity search, recommendation systems, and RAG pipelines."
    ]
    print(summarize(docs, max_sentences=2))
