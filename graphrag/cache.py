import os
import re
import logging
from cachetools import TTLCache

logger = logging.getLogger(__name__)

def is_cache_enabled() -> bool:
    return os.getenv("CACHE_ENABLED", "true").lower() == "true"

def normalize_query(query: str) -> str:
    """Normalize query for cache key generation.
    Strips case, trailing punctuation, and extra whitespace.
    """
    q = query.lower().strip()
    q = re.sub(r'[?!.,;:]+$', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q

class QueryCache:
    """Generic TTL cache with hit/miss tracking."""
    def __init__(self, ttl: int, maxsize: int):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.hits = 0
        self.misses = 0

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key, value):
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def stats(self) -> dict:
        return {
            "size": len(self.cache),
            "maxsize": self.cache.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate_percent": round((self.hits / (self.hits + self.misses) * 100) if (self.hits + self.misses) > 0 else 0, 2)
        }

def get_response_cache() -> QueryCache:
    ttl = int(os.getenv("CACHE_RESPONSE_TTL", 3600))
    maxsize = int(os.getenv("CACHE_RESPONSE_MAX_SIZE", 500))
    return QueryCache(ttl=ttl, maxsize=maxsize)

def get_bundle_cache() -> QueryCache:
    ttl = int(os.getenv("CACHE_BUNDLE_TTL", 1800))
    maxsize = int(os.getenv("CACHE_BUNDLE_MAX_SIZE", 200))
    return QueryCache(ttl=ttl, maxsize=maxsize)

