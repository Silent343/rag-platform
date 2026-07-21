"""Small in-memory per-IP rate limiter for public demo deployments."""

from collections import defaultdict, deque
from threading import Lock
from time import monotonic

from fastapi import HTTPException, Request, status


class RateLimiter:
    """Limits requests by client IP within a rolling time window."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, request: Request) -> None:
        forwarded_for = request.headers.get("x-forwarded-for", "")
        client_ip = forwarded_for.split(",")[0].strip() or (request.client.host if request.client else "unknown")
        now = monotonic()
        with self._lock:
            timestamps = self._requests[client_ip]
            while timestamps and now - timestamps[0] >= self._window_seconds:
                timestamps.popleft()
            if len(timestamps) >= self._max_requests:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Too many requests. Please wait a minute and try again.")
            timestamps.append(now)
