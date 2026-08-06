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

    def test_repeated_violations_trigger_temporary_block(self) -> None:
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(
            1,
            10.0,
            violation_limit=2,
            block_seconds=30.0,
            clock=clock,
        )

        self.assertIsNone(limiter.check("spammer"))
        self.assertEqual(limiter.check("spammer"), 10.0)
        self.assertEqual(limiter.check("spammer"), 30.0)

        clock.now = 20.0
        self.assertEqual(limiter.check("spammer"), 10.0)
        clock.now = 30.1
        self.assertIsNone(limiter.check("spammer"))

    def test_busy_request_violations_also_trigger_block(self) -> None:
        clock = FakeClock()
        limiter = SlidingWindowRateLimiter(
            10,
            60.0,
            violation_limit=3,
            block_seconds=300.0,
            clock=clock,
        )

        self.assertEqual(limiter.violate("busy-spammer"), 0.0)
        self.assertEqual(limiter.violate("busy-spammer"), 0.0)
        self.assertEqual(limiter.violate("busy-spammer"), 300.0)
        self.assertEqual(limiter.check("busy-spammer"), 300.0)


if __name__ == "__main__":
    unittest.main()
