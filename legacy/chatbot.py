import os
import re
import json
import requests

# Colors for terminal styling
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

# Hardcoded fallback key from the user
API_KEY = os.environ.get("GROQ_API_KEY", "Use-your-own-api-key")

GRAPH_FILE = os.path.join(os.path.dirname(__file__), "iitjammu_ee_markdown", "extracted_graph.json")

def load_graph():
    if not os.path.exists(GRAPH_FILE):
        print(f"{C_RED}Error: Graph Database file not found at {GRAPH_FILE}.{C_RESET}")
        print(f"Please run {C_YELLOW}python extract_full_kg.py{C_RESET} first to generate the database!")
        exit(1)
        
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        graph = json.load(f)
    return graph["nodes"], graph["edges"]

def clean_term(text):
    return re.sub(r'[^\w\s]', '', text.lower()).strip()

def search_nodes(query, nodes):
    """Find nodes in the graph that match keyword terms in the query."""
    query_clean = clean_term(query)
    query_words = set(query_clean.split())
    
    # Exclude tiny/common academic stopwords to prevent false matching
    stopwords = {"the", "and", "who", "what", "where", "how", "list", "show", "professor", "student", "faculty", "iit", "jammu", "research", "area", "project", "details", "info"}
    query_words = {w for w in query_words if w not in stopwords and len(w) > 2}
    
    matched = []
    
    for node in nodes:
        node_id = node["id"]
        node_id_clean = clean_term(node_id)
        label = node["label"].lower()
        properties = node["properties"]
        
        # 1. Exact node ID match
        if node_id_clean in query_clean or query_clean in node_id_clean:
            matched.append(node)
            continue
            
        # 2. Match individual significant words
        node_id_words = set(node_id_clean.split())
        sig_matches = node_id_words.intersection(query_words)
        # Exclude tiny words
        sig_matches = {w for w in sig_matches if len(w) > 3}
        if sig_matches:
            matched.append(node)
            continue
            
        # 3. Match other property values (like Email or Designation)
        property_matched = False
        for k, v in properties.items():
            if k == "raw_text":
                continue
            if isinstance(v, str) and len(v) > 3:
                v_clean = clean_term(v)
                if v_clean in query_clean or any(qw in v_clean.split() for qw in query_words if len(qw) > 3):
                    property_matched = True
                    break
        if property_matched:
            matched.append(node)
            
    return matched

def get_fallback_document_context(query, nodes):
    """Fallback search: if no entities match, search raw text of Document nodes for keywords."""
    query_clean = clean_term(query)
    query_words = [w for w in query_clean.split() if len(w) > 3]
    
    best_doc = None
    max_matches = 0
    
    for node in nodes:
        if node["label"] == "Document":
            raw_text = node["properties"].get("raw_text", "").lower()
            matches = sum(1 for w in query_words if w in raw_text)
            if matches > max_matches:
                max_matches = matches
                best_doc = node
                
    if best_doc and max_matches > 0:
        raw_text = best_doc["properties"]["raw_text"]
        # Pull out a relevant chunk of raw text to avoid huge token usage
        # Find first word match location
        first_match_idx = 0
        for w in query_words:
            idx = raw_text.lower().find(w)
            if idx != -1:
                first_match_idx = idx
                break
        
        start = max(0, first_match_idx - 500)
        end = min(len(raw_text), first_match_idx + 2500)
        text_chunk = raw_text[start:end]
        
        return f"\n=== FALLBACK RELEVANT DOCUMENT: {best_doc['id']} ===\n... {text_chunk} ..."
    return ""

def retrieve_subgraph_context(query, nodes, edges):
    matched_nodes = search_nodes(query, nodes)
    
    # If no structured nodes matched, use fallback search in raw document contents
    if not matched_nodes:
        fallback_context = get_fallback_document_context(query, nodes)
        if fallback_context:
            return fallback_context
        return "No matching entities or documents found in the graph database."
        
    # Cap matched nodes to prevent context limit explosion
    matched_nodes = matched_nodes[:8]
    matched_node_ids = {n["id"] for n in matched_nodes}
    
    # Fetch 1-hop relationship edges
    relevant_edges = []
    neighbor_node_ids = set()
    
    for edge in edges:
        src = edge["source"]
        tgt = edge["target"]
        
        if src in matched_node_ids or tgt in matched_node_ids:
            relevant_edges.append(edge)
            neighbor_node_ids.add(src)
            neighbor_node_ids.add(tgt)
            
    # Fetch properties of neighbors
    neighbor_nodes = [n for n in nodes if n["id"] in neighbor_node_ids and n["id"] not in matched_node_ids]
    neighbor_nodes = neighbor_nodes[:15] # Cap neighbors
    
    # Formulate structural Context
    context_lines = []
    
    context_lines.append("=== RETRIEVED GRAPH ENTITIES ===")
    for node in matched_nodes:
        nid = node["id"]
        label = node["label"]
        props = {k: v for k, v in node["properties"].items() if k != "raw_text"}
        context_lines.append(f"- Entity [{label}]: '{nid}' Properties: {json.dumps(props)}")
        
        # If matching a specific Document node, inject its raw content snippet
        if label == "Document" and "raw_text" in node["properties"]:
            snippet = node["properties"]["raw_text"][:2500]
            context_lines.append(f"  [Raw Text Content of {nid}]:\n{snippet}\n")
            
    context_lines.append("\n=== NEIGHBORING ENTITIES (1-HOP) ===")
    for node in neighbor_nodes:
        nid = node["id"]
        label = node["label"]
        props = {k: v for k, v in node["properties"].items() if k != "raw_text"}
        context_lines.append(f"- Entity [{label}]: '{nid}' Properties: {json.dumps(props)}")
        
    context_lines.append("\n=== GRAPH RELATIONSHIPS (TRIPLES) ===")
    for edge in relevant_edges[:40]:
        src = edge["source"]
        tgt = edge["target"]
        etype = edge["type"]
        context_lines.append(f"- Triplet: ('{src}') --[:{etype}]--> ('{tgt}')")
        
    return "\n".join(context_lines)

def call_groq_api(prompt):
    """Direct HTTP REST call to Groq's Mixtral model."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        # Groq returns choices[0].message.content
        return res_json["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling Groq API: {e}"


def run_chatbot():
    print(f"{C_BLUE}{C_BOLD}========================================================================={C_RESET}")
    print(f"{C_GREEN}{C_BOLD}     🤖 Welcome to the IIT Jammu EE GraphRAG Knowledge Chatbot 🤖{C_RESET}")
    print(f"{C_BLUE}========================================================================={C_RESET}")
    print(f"Loading Graph Database...")
    
    nodes, edges = load_graph()
    print(f"{C_CYAN}Success! Loaded {len(nodes)} nodes and {len(edges)} relations.{C_RESET}")
    print("Database holds Faculty, PhD Students, Funded Projects, Patents, and Startups.")
    print("Type your questions below. Type 'exit' or 'quit' to close the chatbot.\n")
    
    while True:
        try:
            query = input(f"{C_BOLD}User: {C_RESET}")
            if query.strip().lower() in ["exit", "quit"]:
                print(f"\n{C_GREEN}Goodbye! Have a great day!{C_RESET}")
                break
                
            if not query.strip():
                continue
                
            print(f"{C_YELLOW}Querying Knowledge Graph Database...{C_RESET}")
            
            # Retrieve subgraph context
            context = retrieve_subgraph_context(query, nodes, edges)
            
            # Formulate full LLM prompt
            prompt = f"""You are a helpful, professional AI chatbot representing the Department of Electrical Engineering at the Indian Institute of Technology Jammu (IIT Jammu).
You have access to a structured Graph Database containing all properties, relationships, and raw document contents parsed directly from the department's website.

Here is the exact structural graph context retrieved from the database related to the user's query:
----------------------------------------
{context}
----------------------------------------

User Question: {query}

Instructions:
1. Ground your answer completely in the retrieved Graph context above.
2. If the context contains relative or absolute links, preserve them in your answer so the user can click them (e.g. [Profile](url) or email addresses).
3. If the graph context doesn't contain the answer, politely state that the database doesn't seem to hold that specific record. Do NOT make up or hallucinate any facts.
4. Keep your answer professional, accurate, and concise.

Provide your final response:"""

            # Call Groq API
            response_text = call_groq_api(prompt)
            
            print(f"\n{C_GREEN}{C_BOLD}Chatbot:{C_RESET}")
            print(response_text)
            print(f"{C_BLUE}-------------------------------------------------------------------------{C_RESET}\n")
            
        except KeyboardInterrupt:
            print(f"\n\n{C_GREEN}Goodbye!{C_RESET}")
            break
        except Exception as e:
            print(f"\n{C_RED}An error occurred: {e}{C_RESET}\n")

if __name__ == "__main__":
    run_chatbot()
