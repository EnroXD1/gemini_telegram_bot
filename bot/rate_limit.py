from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    def __init__(
        self,
        requests: int,
        window_seconds: float,
        *,
        violation_limit: int = 3,
        block_seconds: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._requests = requests
        self._window = window_seconds
        self._violation_limit = max(1, violation_limit)
        self._block_seconds = max(0.0, block_seconds)
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._violations: dict[str, deque[float]] = {}
        self._blocked_until: dict[str, float] = {}
        self._checks = 0

    def check(self, key: str) -> float | None:
        """Register a request; return retry delay if the window is full."""
        now = self._clock()
        blocked_until = self._blocked_until.get(key, 0.0)
        if blocked_until > now:
            return blocked_until - now
        self._blocked_until.pop(key, None)

        cutoff = now - self._window
        events = self._events.setdefault(key, deque())
        while events and events[0] <= cutoff:
            events.popleft()

        self._checks += 1
        if self._checks % 1024 == 0:
            self._remove_inactive_keys(cutoff, except_key=key)

        if len(events) >= self._requests:
            retry_after = max(0.0, self._window - (now - events[0]))
            return self._record_violation(key, now, cutoff, retry_after)

        events.append(now)
        return None

    def violate(self, key: str) -> float:
        """Register an abuse signal such as another request while one is active."""
        now = self._clock()
        blocked_until = self._blocked_until.get(key, 0.0)
        if blocked_until > now:
            return blocked_until - now
        self._blocked_until.pop(key, None)
        return self._record_violation(key, now, now - self._window, 0.0)

    def _record_violation(
        self,
        key: str,
        now: float,
        cutoff: float,
        retry_after: float,
    ) -> float:
        violations = self._violations.setdefault(key, deque())
        while violations and violations[0] <= cutoff:
            violations.popleft()
        violations.append(now)
        if self._block_seconds > 0 and len(violations) >= self._violation_limit:
            self._blocked_until[key] = now + self._block_seconds
            violations.clear()
            return self._block_seconds
        return retry_after

    def _remove_inactive_keys(self, cutoff: float, *, except_key: str) -> None:
        for key, events in list(self._events.items()):
            if key == except_key:
                continue
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self._events.pop(key, None)
                self._violations.pop(key, None)
                if self._blocked_until.get(key, 0.0) <= self._clock():
                    self._blocked_until.pop(key, None)
