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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._requests = requests
        self._window = window_seconds
        self._clock = clock
        self._events: dict[str, deque[float]] = {}
        self._checks = 0

    def check(self, key: str) -> float | None:
        """Register a request; return retry delay if the window is full."""
        now = self._clock()
        cutoff = now - self._window
        events = self._events.setdefault(key, deque())
        while events and events[0] <= cutoff:
            events.popleft()

        self._checks += 1
        if self._checks % 1024 == 0:
            self._remove_inactive_keys(cutoff, except_key=key)

        if len(events) >= self._requests:
            return max(0.0, self._window - (now - events[0]))

        events.append(now)
        return None

    def _remove_inactive_keys(self, cutoff: float, *, except_key: str) -> None:
        for key, events in list(self._events.items()):
            if key == except_key:
                continue
            while events and events[0] <= cutoff:
                events.popleft()
            if not events:
                self._events.pop(key, None)
