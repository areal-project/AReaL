import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pause_continuation_budget import (
    POST_PAUSE_CONTINUATION_TOKEN_QUOTA,
    get_post_pause_request_max_tokens,
)


class PauseContinuationBudgetTest(unittest.TestCase):
    def test_uses_remaining_context_when_no_quota(self):
        self.assertEqual(get_post_pause_request_max_tokens(12000, None), 12000)

    def test_caps_with_remaining_quota(self):
        self.assertEqual(get_post_pause_request_max_tokens(12000, 3000), 3000)
        self.assertEqual(get_post_pause_request_max_tokens(2000, 3000), 2000)

    def test_zero_when_context_or_quota_exhausted(self):
        self.assertEqual(get_post_pause_request_max_tokens(0, 3000), 0)
        self.assertEqual(get_post_pause_request_max_tokens(2000, 0), 0)
        self.assertEqual(get_post_pause_request_max_tokens(2000, -1), 0)

    def test_default_quota_is_4096(self):
        self.assertEqual(POST_PAUSE_CONTINUATION_TOKEN_QUOTA, 4096)


if __name__ == "__main__":
    unittest.main()
