from graphrag.rules_retriever import RulesRetriever

retriever = RulesRetriever()

queries = [
    "what is the minimum CGPA required for minor in B.Tech?",
    "what are the milestones for a Ph.D. candidate?",
    "grading system and grade point values",
    "can I drop a semester as a B.Tech student?"
]

for q in queries:
    print(f"\n============================================================\nQUERY: '{q}'\n")
    results = retriever.retrieve(q)
    context = retriever.generate_context(results)
    print(context)
