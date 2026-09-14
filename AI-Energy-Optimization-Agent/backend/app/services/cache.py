import threading
import time

_cache = {}
_lock = threading.Lock()
_default_ttl = 600


def cached(key: str, fn, ttl: int = _default_ttl):
    now = time.time()
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            value, expires = hit
            if expires > now:
                return value
    value = fn()
    with _lock:
        _cache[key] = (value, now + ttl)
    return value


def invalidate_cache():
    with _lock:
        _cache.clear()