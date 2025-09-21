#!/usr/bin/env python3
"""
Single query test
"""

import sys
from retriever import OllamaRetriever

def query(search_text):
    retriever = OllamaRetriever()
    results = retriever.search(search_text, top_k=5)
    
    print(f"🔍 Query: '{search_text}'")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        print(f"\n📄 Result {i} (Similarity: {result['score']:.4f})")
        print(f"📝 Text: {result['text']}")
        print("-" * 40)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = input("Enter your search query: ")
    
    query(query_text)