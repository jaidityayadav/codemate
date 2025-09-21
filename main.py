from agent import ResearchAgent

if __name__ == "__main__":
    agent = ResearchAgent()
    print("🔎 Deep Researcher Agent Ready")

    while True:
        query = input("\nEnter your query (or 'exit'): ")
        if query.lower() == "exit":
            break
        response = agent.run(query)
        print("\n📑 Research Result:\n", response)
