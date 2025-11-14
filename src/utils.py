# src/utils.py
"""
Shared utilities: caching, transforms, etc.
"""

import hashlib

# Example: simple in-memory cache for rendered images
class SimpleCache:
    def __init__(self):
        self._cache = {}

    def get(self, key):
        return self._cache.get(key)

    def set(self, key, value):
        self._cache[key] = value

    def clear(self):
        self._cache.clear()

def make_cache_key(*args, **kwargs):
    m = hashlib.sha256()
    m.update(str(args).encode())
    m.update(str(kwargs).encode())
    return m.hexdigest()
