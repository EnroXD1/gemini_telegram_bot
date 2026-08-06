import unittest

from bot.rate_limit import SlidingWindowRateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class RateLimiterTests(unittest.TestCase):
    def test_limit_and_window_expiration(self) -> None:
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(2, 10.0, clock=clock)
        self.assertIsNone(limiter.check("user"))
        self.assertIsNone(limiter.check("user"))
        self.assertEqual(limiter.check("user"), 10.0)
        clock.now = 10.1
        self.assertIsNone(limiter.check("user"))

    def test_keys_are_independent(self) -> None:
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(1, 10.0, clock=clock)
        self.assertIsNone(limiter.check("a"))
        self.assertIsNone(limiter.check("b"))


if __name__ == "__main__":
    unittest.main()
