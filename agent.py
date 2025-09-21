from retriever import OllamaRetriever

class ResearchAgent:
    def __init__(self):
        self.retriever = OllamaRetriever()
        print("🤖 Research Agent initialized with Ollama embeddings")

    def run(self, query: str):
        print(f"🔍 Processing query: {query}")
        
        # Get relevant documents
        results = self.retriever.search(query, top_k=5)
        
        if not results:
            return "❌ No relevant documents found for your query."
        
        # Format the response
        reasoning = f"📋 **Query**: {query}\n\n"
        reasoning += f"🔍 **Found {len(results)} relevant document chunks:**\n\n"
        
        for i, result in enumerate(results, 1):
            score = result['score']
            text = result['text']
            
            reasoning += f"**📄 Result {i}** (Relevance: {score:.3f})\n"
            reasoning += f"{text}\n\n"
            reasoning += "---\n\n"
        
        # Add simple reasoning
        reasoning += "🧠 **Analysis**: "
        reasoning += f"Based on the retrieved documents, the information about '{query}' "
        reasoning += "shows relevant content from your knowledge base. "
        reasoning += f"The top result has a relevance score of {results[0]['score']:.3f}."
        
        return reasoning
