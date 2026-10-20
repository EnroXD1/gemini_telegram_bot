from __future__ import annotations

import time
from collections.abc import Callable


class KeyPool:
    """Sticky until 429, then cyclic failover. Never stringify credentials in logs."""

    def __init__(self, keys: tuple[str, ...], *, clock: Callable[[], float] = time.monotonic):
        self._keys = tuple(dict.fromkeys(keys))
        self._clock = clock
        self._cursor: dict[str, int] = {}
        self._cooldowns: dict[tuple[str, int], float] = {}
        self._probes: set[tuple[str, int]] = set()

    def __len__(self) -> int:
        return len(self._keys)

    def key(self, index: int) -> str:
        return self._keys[index]

    def acquire(self, model: str, tried: set[int]) -> int | None:
        now = self._clock()
        for offset in range(len(self)):
            index = (self._cursor.get(model, 0) + offset) % len(self)
            tag = (model, index)
            if index in tried or tag in self._probes or self._cooldowns.get(tag, 0) > now:
                continue
            self._cursor[model] = index
            if tag in self._cooldowns:
                self._probes.add(tag)  # one recovery request per cooled credential/model
            return index
        return None

    def pause(self, index: int, model: str, seconds: float) -> None:
        tag = (model, index)
        self._cooldowns[tag] = max(self._cooldowns.get(tag, 0), self._clock() + max(1, seconds))
        self._cursor[model] = (index + 1) % len(self)

    def release(self, index: int, model: str, *, success: bool = False) -> None:
        tag = (model, index)
        # Do not erase a newer 429 from another, already in-flight request.
        if success and self._cooldowns.get(tag, 0) <= self._clock():
            self._cooldowns.pop(tag, None)
        self._probes.discard(tag)

    def available(self, model: str) -> int:
        return sum(
            (model, i) not in self._probes
            and self._cooldowns.get((model, i), 0) <= self._clock()
            for i in range(len(self))
        )

    def retry_in(self, model: str) -> float:
        if not self or self.available(model):
            return 0.0
        now = self._clock()
        return min(max(1.0, self._cooldowns.get((model, i), 0) - now) for i in range(len(self)))
