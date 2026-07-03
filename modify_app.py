import re

with open("app.py", "r") as f:
    content = f.read()

setup_block = """
    def _finalize_response(resp_dict):
        if response_cache and resp_dict.get("information_available"):
            response_cache.set(cache_key, resp_dict)
        return jsonify(resp_dict)

    from graphrag.cache import normalize_query
    cache_key = normalize_query(query) if response_cache else None

    if response_cache:
        cached_resp = response_cache.get(cache_key)
        if cached_resp:
            logger.info(f"[CACHE HIT] L1 Response Cache for query: {query}")
            return jsonify(cached_resp)

    try:"""

# Only replace the first occurrence of `    try:` in chat()
# (We know chat() comes right before the first `    try:`)
first_try_idx = content.find("    try:\n        start = time.time()")
content = content[:first_try_idx] + setup_block + content[first_try_idx + 8:]

start_idx = content.find(setup_block)
end_idx = content.find("def llm_status():")

sub_content = content[start_idx:end_idx]
sub_content = sub_content.replace("return jsonify({", "return _finalize_response({")

content = content[:start_idx] + sub_content + content[end_idx:]

admin_routes = '''
@app.route("/api/cache/stats", methods=["GET"])
def cache_stats():
    stats = {}
    if response_cache:
        stats["L1_response_cache"] = response_cache.stats()
    else:
        stats["L1_response_cache"] = "disabled"
        
    stats["L2_bundle_caches"] = {}
    for code, ret in retrievers.items():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            stats["L2_bundle_caches"][code] = ret.bundle_cache.stats()
    for code, ret in section_retrievers.items():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            stats["L2_bundle_caches"][code] = ret.bundle_cache.stats()
            
    return jsonify(stats)

@app.route("/api/cache/clear", methods=["POST"])
def cache_clear():
    cleared = 0
    if response_cache:
        response_cache.clear()
        cleared += 1
    for ret in retrievers.values():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            ret.bundle_cache.clear()
            cleared += 1
    for ret in section_retrievers.values():
        if hasattr(ret, "bundle_cache") and ret.bundle_cache:
            ret.bundle_cache.clear()
            cleared += 1
    return jsonify({"ok": True, "message": f"Cleared {cleared} caches."})

'''
content = content.replace('if __name__ == "__main__":', admin_routes + 'if __name__ == "__main__":')

with open("app.py", "w") as f:
    f.write(content)
