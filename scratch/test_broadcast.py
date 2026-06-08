import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

def main():
    app.init_app()
    mr = app.multi_retriever
    
    q = "Who works on RF Antennas and microwave devices?"
    print("\n" + "="*50)
    print(f"QUERY: {q}")
    print("="*50)
    
    focus_terms = mr._extract_query_focus_terms(q)
    print(f"Focus terms: {focus_terms}")
    
    for code in ["civil_engineering", "mechanical_engineering", "materials-engineering", "ee", "physics"]:
        retriever = mr.retrievers.get(code)
        bundle = retriever.retrieve_bundle(
            q, local_top_k=4, vector_top_k=3, global_top_k=1, max_context_words=2000
        )
        prov = bundle.get("provenance", {})
        
        # 1. Direct
        is_direct = prov.get("route") in ("direct_graph", "direct_graph_multi") or prov.get("graph", {}).get("direct", False)
        
        # 2. Focus terms
        context = bundle.get("context", "")
        context_words = set()
        for word in re_findall_words(context):
            norm_word = mr._normalize_token(word)
            if norm_word:
                context_words.add(norm_word)
        matched_terms = [term for term in focus_terms if term in context_words]
        
        # 3. Graph score
        graph_items = prov.get("graph", {}).get("items", 0)
        graph_avg = prov.get("graph", {}).get("avg_score", 0.0)
        
        # 4. Vector score
        vector_items = prov.get("vector", {}).get("items", 0)
        vector_avg = prov.get("vector", {}).get("avg_score", 0.0)
        is_vector_high = vector_items > 0 and vector_avg >= 0.45
        
        is_relevant = mr._is_bundle_relevant(q, bundle, is_topic=True)
        
        print(f"Department: {code}")
        print(f"  Is Direct: {is_direct}")
        print(f"  Matched terms: {matched_terms}")
        print(f"  Is Vector High: {is_vector_high} (items={vector_items}, avg={vector_avg})")
        print(f"  Overall Relevant: {is_relevant}")

def re_findall_words(text):
    import re
    return re.findall(r"[A-Za-z0-9]+", text.lower())

if __name__ == "__main__":
    main()
