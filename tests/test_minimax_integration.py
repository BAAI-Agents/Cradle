"""Integration tests for MiniMax LLM provider.

These tests make real API calls to the MiniMax API.
They require the MINIMAX_API_KEY environment variable to be set.
Skip if the key is not available.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API."""

    def _make_provider(self, model="MiniMax-M2.7"):
        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "comp_model": model,
            "base_url": "https://api.minimax.io/v1",
        }
        provider.init_provider(config_dict)
        return provider

    def test_simple_completion(self):
        provider = self._make_provider()
        messages = [
            {"role": "user", "content": "Reply with exactly: HELLO_MINIMAX"}
        ]
        # M2.7 uses thinking tokens that consume max_tokens budget, so allow enough room
        result, info = provider.create_completion(messages, temperature=0.01, max_tokens=512)
        self.assertIn("HELLO_MINIMAX", result)
        self.assertGreater(info["total_tokens"], 0)

    def test_completion_with_system_message(self):
        provider = self._make_provider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply with exactly: SYSTEM_TEST_OK"},
        ]
        result, info = provider.create_completion(messages, temperature=0.01, max_tokens=512)
        self.assertIn("SYSTEM_TEST_OK", result)

    def test_completion_with_m25_model(self):
        provider = self._make_provider(model="MiniMax-M2.5")
        messages = [
            {"role": "user", "content": "Reply with exactly: M25_OK"}
        ]
        result, info = provider.create_completion(messages, temperature=0.01, max_tokens=512)
        self.assertIn("M25_OK", result)


if __name__ == "__main__":
    unittest.main()
