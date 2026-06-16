import os
import pickle
import sys
sys.path.append("/home/c3i/chatbot")
from graphrag.section_retriever import SectionRetriever
from graphrag.section_kg_builder import SectionKGBuilder
from graphrag.embeddings import EmbeddingEngine
from departments import get_section_data_dir

def test_query(retriever, query):
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)
    bundle = retriever.retrieve_bundle(query)
    print(f"Route: {bundle['provenance']['route']}")
    print(f"Answerability: {bundle['answerability']}")
    print(f"Word count: {len(bundle['context'].split()) if bundle['context'] else 0}")
    print("\nCONTEXT PREVIEW:")
    lines = bundle['context'].split('\n')
    for line in lines[:20]:
        print(line)
    if len(lines) > 20:
        print("...")
    print("\n")

def main():
    data_dir = get_section_data_dir("academics")
    
    # Load graph and chunks
    print("Loading Academics section data...")
    graph, chunks = SectionKGBuilder.load(data_dir)
    
    # Initialize Embedding Engine
    print("Loading Embedding Engine...")
    embeddings = EmbeddingEngine()
    embeddings.load(data_dir)
    
    retriever = SectionRetriever(
        section_code="academics",
        graph=graph,
        chunks=chunks,
        embedding_engine=embeddings
    )
    
    queries = [
        "How many credits do a Civil Engineering student require to graduate",
        "How can someone get institute silver medal",
        "Who got the institute silver medal"
    ]
    
    for q in queries:
        test_query(retriever, q)

if __name__ == "__main__":
    main()
