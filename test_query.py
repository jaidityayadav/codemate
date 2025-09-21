#!/usr/bin/env python3
"""
Simple test script for querying your vector database
"""

from retriever import OllamaRetriever

def test_retrieval():
    print("🚀 Testing Vector Database Retrieval")
    print("=" * 50)
    
    # Initialize retriever
    try:
        retriever = OllamaRetriever()
        print("✅ Retriever initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing retriever: {e}")
        return
    
    # Test queries
    test_queries = [
        "What is the main topic?",
        "Tell me about the key points",
        "What are the important concepts?",
        "Summarize the content"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        try:
            results = retriever.search(query, top_k=3)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n📄 Result {i} (Score: {result['score']:.4f}):")
                    print(f"   {result['text'][:200]}...")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

if __name__ == "__main__":
    test_retrieval()