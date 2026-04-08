"""
Task registry for CodeReviewEnv.

Each task is a dict with:
  pr_title            : str
  pr_description      : str
  diff                : str  — the code diff the agent reviews
  ground_truth_issues : list of issue dicts
  expected_decision   : "approve" | "request_changes"

Each ground-truth issue dict:
  id              : str   — stable unique identifier
  severity        : str   — critical / high / medium / low
  match_keywords  : list  — words that should appear in a correct comment
  fix_keywords    : list  — words expected in a good fix suggestion
"""

# ---------------------------------------------------------------------------
# TASK 1 — EASY
# One obvious null-pointer / None-dereference bug.
# An agent needs to find it, classify it correctly (high), suggest a guard.
# ---------------------------------------------------------------------------

EASY_DIFF = '''\
diff --git a/user_service.py b/user_service.py
--- a/user_service.py
+++ b/user_service.py
@@ -1,20 +1,34 @@
 import logging
 from database import get_user_by_id
 
+logger = logging.getLogger(__name__)
+
 def send_welcome_email(user_id: int) -> bool:
     """Send a welcome email to a newly registered user."""
-    user = get_user_by_id(user_id)
+    user = get_user_by_id(user_id)   # can return None if not found
+    email_address = user.email       # BUG: no None-check on user
+    subject = "Welcome to our platform!"
+    body = f"Hi {user.first_name}, thanks for joining!"
+    return _send(email_address, subject, body)
+
+def get_user_profile(user_id: int) -> dict:
+    """Return a dict of public profile fields."""
+    user = get_user_by_id(user_id)
     if user is None:
-        return False
-    return _send(user.email, "Welcome!", f"Hi {user.first_name}")
+        return {}
+    return {
+        "id": user.id,
+        "name": user.first_name + " " + user.last_name,
+        "joined": user.created_at.isoformat(),
+    }
 
-def delete_account(user_id: int) -> None:
+def delete_account(user_id: int, confirm: bool = False) -> None:
     """Permanently delete a user account."""
-    user = get_user_by_id(user_id)
-    user.delete()
+    if not confirm:
+        raise ValueError("Must pass confirm=True to delete an account.")
+    user = get_user_by_id(user_id)
+    if user is None:
+        logger.warning("delete_account called for unknown user %d", user_id)
+        return
+    user.delete()
'''

EASY_TASK = {
    "pr_title": "feat: add welcome email + profile endpoint",
    "pr_description": (
        "Adds send_welcome_email(), get_user_profile(), and improves delete_account() "
        "with a confirmation guard. Straightforward CRUD helpers."
    ),
    "diff": EASY_DIFF,
    "expected_decision": "request_changes",
    "ground_truth_issues": [
        {
            "id": "null-deref-send-welcome",
            "severity": "high",
            "match_keywords": ["null", "none", "user", "email", "send_welcome", "dereference", "check"],
            "fix_keywords": ["none", "guard", "is None", "check", "return False", "AttributeError"],
            "description": "send_welcome_email() does not check whether get_user_by_id() returned None before accessing user.email and user.first_name, causing AttributeError on unknown user IDs.",
        }
    ],
}

# ---------------------------------------------------------------------------
# TASK 2 — MEDIUM
# Two bugs: an off-by-one logic error AND a missing thread-safety guard.
# ---------------------------------------------------------------------------

MEDIUM_DIFF = '''\
diff --git a/rate_limiter.py b/rate_limiter.py
--- a/rate_limiter.py
+++ b/rate_limiter.py
@@ -1,30 +1,55 @@
+import time
+import threading
 from collections import deque
-from datetime import datetime, timedelta
 
 class RateLimiter:
-    """Allow at most `max_calls` per `period` seconds."""
+    """Allow at most `max_calls` per `period` seconds per key."""
 
-    def __init__(self, max_calls: int, period: float):
+    def __init__(self, max_calls: int, period: float, strict: bool = False):
         self.max_calls = max_calls
         self.period = period
-        self._calls: deque = deque()
+        self.strict = strict
+        self._buckets: dict[str, deque] = {}
+        # No lock — concurrent callers from different threads share _buckets
 
-    def is_allowed(self) -> bool:
-        now = datetime.utcnow()
-        cutoff = now - timedelta(seconds=self.period)
-        while self._calls and self._calls[0] < cutoff:
-            self._calls.popleft()
-        if len(self._calls) >= self.max_calls:
+    def is_allowed(self, key: str = "default") -> bool:
+        now = time.monotonic()
+        cutoff = now - self.period
+        bucket = self._buckets.setdefault(key, deque())
+        # Evict old timestamps
+        while bucket and bucket[0] <= cutoff:   # BUG: should be < not <=, drops calls exactly at boundary
+            bucket.popleft()
+        if len(bucket) >= self.max_calls:       # correctly >= here
             return False
-        self._calls.append(now)
+        bucket.append(now)
         return True
 
-    def reset(self) -> None:
-        self._calls.clear()
+    def reset(self, key: str = "default") -> None:
+        bucket = self._buckets.get(key)
+        if bucket is not None:
+            bucket.clear()
+
+    def stats(self, key: str = "default") -> dict:
+        bucket = self._buckets.get(key, deque())
+        return {"key": key, "call_count": len(bucket), "max_calls": self.max_calls}
 
-def get_limiter(name: str, config: dict) -> RateLimiter:
-    _cache: dict = {}
-    if name not in _cache:          # BUG: _cache is local — always empty
-        _cache[name] = RateLimiter(**config)
-    return _cache[name]
+_limiter_cache: dict[str, RateLimiter] = {}
+
+def get_limiter(name: str, config: dict) -> RateLimiter:
+    if name not in _limiter_cache:
+        _limiter_cache[name] = RateLimiter(**config)
+    return _limiter_cache[name]
'''

MEDIUM_TASK = {
    "pr_title": "refactor: per-key rate limiter with monotonic clock",
    "pr_description": (
        "Rewrites RateLimiter to support per-key bucketing and switches to "
        "time.monotonic(). Also fixes the stale local-cache bug in get_limiter(). "
        "Should be thread-safe for our async worker pool."
    ),
    "diff": MEDIUM_DIFF,
    "expected_decision": "request_changes",
    "ground_truth_issues": [
        {
            "id": "off-by-one-boundary",
            "severity": "medium",
            "match_keywords": ["boundary", "off-by-one", "cutoff", "<=", "less than", "equal", "evict", "timestamp"],
            "fix_keywords": ["<", "strictly less", "boundary", "off-by-one", "exclude"],
            "description": (
                "The eviction condition `bucket[0] <= cutoff` incorrectly drops calls that "
                "happened exactly at the window boundary, causing under-counting and allowing "
                "slightly more requests than max_calls in edge cases."
            ),
        },
        {
            "id": "thread-safety-missing-lock",
            "severity": "high",
            "match_keywords": ["thread", "lock", "race", "concurrent", "safe", "_buckets", "mutex"],
            "fix_keywords": ["lock", "threading.Lock", "RLock", "thread-safe", "concurrent", "synchronized"],
            "description": (
                "RateLimiter._buckets is a shared mutable dict with no lock. Concurrent calls "
                "from multiple threads can corrupt the deque or cause KeyError via a TOCTOU race "
                "in setdefault + append."
            ),
        },
    ],
}

# ---------------------------------------------------------------------------
# TASK 3 — HARD
# Two issues: SQL injection via f-string, AND an unbounded memory accumulation.
# Requires recognising security + performance problems simultaneously.
# ---------------------------------------------------------------------------

HARD_DIFF = '''\
diff --git a/report_service.py b/report_service.py
--- a/report_service.py
+++ b/report_service.py
@@ -1,40 +1,95 @@
+import hashlib
+import json
 import logging
-from db import execute_query
+from typing import Any
+from db import execute_query, execute_query_raw
 
 logger = logging.getLogger(__name__)
 
-_REPORT_CACHE: dict = {}
+_REPORT_CACHE: dict[str, Any] = {}   # module-level, never evicted
 
-def get_report(report_type: str, filters: dict) -> dict:
-    cache_key = f"{report_type}:{sorted(filters.items())}"
+def _cache_key(report_type: str, filters: dict) -> str:
+    return hashlib.md5(f"{report_type}{json.dumps(filters, sort_keys=True)}".encode()).hexdigest()
+
+def get_report(report_type: str, filters: dict, use_cache: bool = True) -> dict:
+    """Fetch a named report, optionally from cache."""
+    key = _cache_key(report_type, filters)
+    if use_cache and key in _REPORT_CACHE:
+        return _REPORT_CACHE[key]
+
+    result = _run_report_query(report_type, filters)
+    _REPORT_CACHE[key] = result   # stored forever — no TTL, no max size
+    return result
+
+def _run_report_query(report_type: str, filters: dict) -> dict:
+    """Build and execute the SQL for the given report type."""
+    # Validate report type from whitelist
     allowed = {"sales", "inventory", "users"}
     if report_type not in allowed:
         raise ValueError(f"Unknown report type: {report_type}")
-    if cache_key in _REPORT_CACHE:
-        return _REPORT_CACHE[cache_key]
-    sql = _build_query(report_type, filters)
-    result = execute_query(sql)
-    _REPORT_CACHE[cache_key] = result
-    return result
 
-def _build_query(report_type: str, filters: dict) -> str:
-    parts = [f"SELECT * FROM {report_type}_view"]
-    conditions = [f"{k} = '{v}'" for k, v in filters.items()]
+    parts = [f"SELECT * FROM {report_type}_view"]   # report_type is whitelisted
+    conditions = []
+    for k, v in filters.items():
+        # BUG: f-string interpolation of filter keys/values — SQL injection via filter keys
+        conditions.append(f"{k} = '{v}'")
     if conditions:
         parts.append("WHERE " + " AND ".join(conditions))
-    return " ".join(parts)
+    sql = " ".join(parts)
+    logger.debug("Executing report SQL: %s", sql)
+    return execute_query_raw(sql)   # raw execution, no parameterisation
 
 def export_report_csv(report_type: str, filters: dict, path: str) -> str:
-    data = get_report(report_type, filters)
-    with open(path, "w") as f:
-        ...
-    return path
+    """Export report data to a CSV file at `path`."""
+    data = get_report(report_type, filters)
+    rows = data.get("rows", [])
+    if not rows:
+        return path
+    headers = list(rows[0].keys())
+    with open(path, "w", newline="") as f:
+        import csv
+        writer = csv.DictWriter(f, fieldnames=headers)
+        writer.writeheader()
+        writer.writerows(rows)
+    return path
+
+def invalidate_cache(report_type: str = None) -> int:
+    """Remove cached entries. Pass report_type to be selective, or None for all."""
+    removed = 0
+    if report_type is None:
+        removed = len(_REPORT_CACHE)
+        _REPORT_CACHE.clear()
+    else:
+        keys_to_remove = [k for k in _REPORT_CACHE]  # can't selectively remove by type (opaque hash keys)
+        for k in keys_to_remove:
+            del _REPORT_CACHE[k]
+            removed += 1
+    return removed
'''

HARD_TASK = {
    "pr_title": "perf: MD5-keyed report cache + CSV export",
    "pr_description": (
        "Switches cache key to MD5 hash for consistency, adds export_report_csv(), "
        "and adds an invalidate_cache() helper. The report_type whitelist prevents "
        "arbitrary table access. Performance should improve significantly for repeated queries."
    ),
    "diff": HARD_DIFF,
    "expected_decision": "request_changes",
    "ground_truth_issues": [
        {
            "id": "sql-injection-filter-keys",
            "severity": "critical",
            "match_keywords": ["sql", "injection", "filter", "key", "interpolat", "parameteris", "escape", "f-string", "conditions"],
            "fix_keywords": ["parameteris", "placeholder", "?", "%s", "bind", "sanitise", "whitelist", "prepared"],
            "description": (
                "Filter keys and values are interpolated directly into SQL via an f-string without "
                "sanitisation or parameterised queries. An attacker can inject arbitrary SQL through "
                "a crafted filter key (e.g. `\"1=1; DROP TABLE users; --\"`)."
            ),
        },
        {
            "id": "unbounded-cache-memory-leak",
            "severity": "high",
            "match_keywords": ["cache", "memory", "unbounded", "evict", "ttl", "leak", "grow", "size", "limit"],
            "fix_keywords": ["ttl", "lru", "maxsize", "evict", "expire", "bounded", "limit", "functools.lru_cache"],
            "description": (
                "_REPORT_CACHE is a module-level dict that is never evicted (no TTL, no max-size). "
                "In a long-running service, unique filter combinations will cause unbounded memory "
                "growth, eventually leading to OOM crashes."
            ),
        },
    ],
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: dict[str, dict] = {
    "easy":   EASY_TASK,
    "medium": MEDIUM_TASK,
    "hard":   HARD_TASK,
}
