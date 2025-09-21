from retriever import Retriever

class ResearchAgent:
    def __init__(self):
        self.retriever = Retriever()

    def run(self, query: str):
        docs = self.retriever.search(query)
        # Multi-step reasoning (very simplified)
        reasoning = f"Query: {query}\n\nRelevant Docs:\n"
        for i, d in enumerate(docs):
            reasoning += f"{i+1}. {d[:300]}...\n\n"
        return reasoning
