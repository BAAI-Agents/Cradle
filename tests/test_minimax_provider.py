"""Unit tests for MiniMax LLM provider."""
import os
import json
import unittest
from unittest.mock import patch, MagicMock

import sys
# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestClampTemperature(unittest.TestCase):
    """Test temperature clamping utility."""

    def test_clamp_zero_temperature(self):
        from cradle.provider.llm.minimax import _clamp_temperature
        self.assertEqual(_clamp_temperature(0.0), 0.01)

    def test_clamp_negative_temperature(self):
        from cradle.provider.llm.minimax import _clamp_temperature
        self.assertEqual(_clamp_temperature(-1.0), 0.01)

    def test_clamp_high_temperature(self):
        from cradle.provider.llm.minimax import _clamp_temperature
        self.assertEqual(_clamp_temperature(2.0), 1.0)

    def test_normal_temperature_passes_through(self):
        from cradle.provider.llm.minimax import _clamp_temperature
        self.assertEqual(_clamp_temperature(0.5), 0.5)

    def test_max_temperature_boundary(self):
        from cradle.provider.llm.minimax import _clamp_temperature
        self.assertEqual(_clamp_temperature(1.0), 1.0)

    def test_min_boundary_temperature(self):
        from cradle.provider.llm.minimax import _clamp_temperature
        self.assertEqual(_clamp_temperature(0.01), 0.01)


class TestMiniMaxProviderConfig(unittest.TestCase):
    """Test MiniMax provider configuration parsing."""

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_parse_config_from_dict(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "emb_model": "text-embedding-ada-002",
            "comp_model": "MiniMax-M2.7",
            "base_url": "https://api.minimax.io/v1",
        }

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider.init_provider(config_dict)

        self.assertEqual(provider.llm_model, "MiniMax-M2.7")
        self.assertEqual(provider.embedding_model, "text-embedding-ada-002")
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.minimax.io/v1"
        )

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_parse_config_default_base_url(self, mock_openai_cls):
        mock_openai_cls.return_value = MagicMock()

        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "comp_model": "MiniMax-M2.5",
        }

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider.init_provider(config_dict)

        self.assertEqual(provider.llm_model, "MiniMax-M2.5")
        mock_openai_cls.assert_called_once_with(
            api_key="test-key", base_url="https://api.minimax.io/v1"
        )

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_default_retries(self, mock_openai_cls):
        from cradle.provider.llm.minimax import MiniMaxProvider
        provider = MiniMaxProvider()
        self.assertEqual(provider.retries, 5)


class TestMiniMaxProviderCompletion(unittest.TestCase):
    """Test MiniMax provider completion methods."""

    def _make_provider(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "comp_model": "MiniMax-M2.7",
            "base_url": "https://api.minimax.io/v1",
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider.init_provider(config_dict)

        return provider, mock_client

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_create_completion_success(self, mock_openai_cls):
        provider, mock_client = self._make_provider(mock_openai_cls)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello from MiniMax"
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 5
        mock_response.usage.total_tokens = 15
        mock_response.system_fingerprint = "fp_abc"
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result, info = provider.create_completion(messages, temperature=0.7)

        self.assertEqual(result, "Hello from MiniMax")
        self.assertEqual(info["total_tokens"], 15)

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_create_completion_strips_think_tags(self, mock_openai_cls):
        provider, mock_client = self._make_provider(mock_openai_cls)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "<think>Let me reason...</think>\nThe answer is 42."
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        mock_response.usage.total_tokens = 30
        mock_response.system_fingerprint = None
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "What is the answer?"}]
        result, info = provider.create_completion(messages, temperature=0.5)

        self.assertEqual(result, "The answer is 42.")
        self.assertNotIn("<think>", result)

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_create_completion_clamps_temperature(self, mock_openai_cls):
        provider, mock_client = self._make_provider(mock_openai_cls)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 6
        mock_response.system_fingerprint = None
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        provider.create_completion(messages, temperature=0.0)

        call_args = mock_client.chat.completions.create.call_args
        self.assertAlmostEqual(call_args.kwargs.get("temperature", call_args[1].get("temperature")), 0.01)

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_create_completion_uses_default_model(self, mock_openai_cls):
        provider, mock_client = self._make_provider(mock_openai_cls)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 6
        mock_response.system_fingerprint = None
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        provider.create_completion(messages, temperature=0.5)

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs.get("model", call_args[1].get("model")), "MiniMax-M2.7")

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_create_completion_custom_model(self, mock_openai_cls):
        provider, mock_client = self._make_provider(mock_openai_cls)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 6
        mock_response.system_fingerprint = None
        mock_client.chat.completions.create.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        provider.create_completion(messages, model="MiniMax-M2.5", temperature=0.5)

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args.kwargs.get("model", call_args[1].get("model")), "MiniMax-M2.5")


class TestMiniMaxProviderEmbedding(unittest.TestCase):
    """Test MiniMax provider embedding methods."""

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_embed_query(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "emb_model": "text-embedding-ada-002",
            "comp_model": "MiniMax-M2.7",
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider.init_provider(config_dict)

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_client.embeddings.create.return_value = mock_response

        result = provider.embed_query("Hello")
        self.assertEqual(result, [0.1, 0.2, 0.3])

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_embed_documents(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "emb_model": "text-embedding-ada-002",
            "comp_model": "MiniMax-M2.7",
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider.init_provider(config_dict)

        mock_emb1 = MagicMock()
        mock_emb1.embedding = [0.1, 0.2]
        mock_emb2 = MagicMock()
        mock_emb2.embedding = [0.3, 0.4]
        mock_response = MagicMock()
        mock_response.data = [mock_emb1, mock_emb2]
        mock_client.embeddings.create.return_value = mock_response

        result = provider.embed_documents(["Hello", "World"])
        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])

    @patch("cradle.provider.llm.minimax.OpenAI")
    def test_get_embedding_dim(self, mock_openai_cls):
        mock_openai_cls.return_value = MagicMock()

        from cradle.provider.llm.minimax import MiniMaxProvider

        provider = MiniMaxProvider()
        config_dict = {
            "key_var": "MINIMAX_API_KEY",
            "emb_model": "text-embedding-ada-002",
            "comp_model": "MiniMax-M2.7",
        }
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key"}):
            provider.init_provider(config_dict)

        self.assertEqual(provider.get_embedding_dim(), 1536)


class TestLLMFactoryMiniMax(unittest.TestCase):
    """Test LLM factory recognizes MiniMax config."""

    @patch("cradle.provider.llm.minimax.OpenAI")
    @patch("cradle.provider.llm.openai.OpenAI")
    def test_factory_creates_minimax_provider(self, mock_openai_openai, mock_minimax_openai):
        mock_openai_openai.return_value = MagicMock()
        mock_minimax_openai.return_value = MagicMock()

        from cradle.provider.llm.llm_factory import LLMFactory
        from cradle.provider.llm.minimax import MiniMaxProvider

        # Reset singleton for testing
        LLMFactory._instances = {}

        factory = LLMFactory()

        # The factory matches by checking if "minimax" is in the config path string
        minimax_config_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "minimax_config.json"
        )
        openai_config_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "openai_config.json"
        )

        # Create a temporary openai_config.json for the embed provider
        openai_cfg = {
            "key_var": "OA_OPENAI_KEY",
            "emb_model": "text-embedding-ada-002",
            "comp_model": "gpt-4o",
            "is_azure": False,
        }
        os.makedirs(os.path.dirname(openai_config_path), exist_ok=True)
        if not os.path.exists(openai_config_path):
            with open(openai_config_path, "w") as f:
                json.dump(openai_cfg, f)

        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key", "OA_OPENAI_KEY": "test-key"}):
            llm_provider, embed_provider = factory.create(
                minimax_config_path, openai_config_path
            )

        self.assertIsInstance(llm_provider, MiniMaxProvider)


class TestMiniMaxConfigFile(unittest.TestCase):
    """Test the MiniMax config JSON file."""

    def test_config_file_valid_json(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "minimax_config.json"
        )
        with open(config_path) as f:
            config = json.load(f)

        self.assertEqual(config["key_var"], "MINIMAX_API_KEY")
        self.assertEqual(config["comp_model"], "MiniMax-M2.7")
        self.assertEqual(config["base_url"], "https://api.minimax.io/v1")
        self.assertIn("emb_model", config)

    def test_config_has_required_fields(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "conf", "minimax_config.json"
        )
        with open(config_path) as f:
            config = json.load(f)

        required = ["key_var", "comp_model"]
        for field in required:
            self.assertIn(field, config, f"Missing required field: {field}")


class TestThinkTagStripping(unittest.TestCase):
    """Test think tag stripping in responses."""

    def test_strip_simple_think_tag(self):
        import re
        msg = "<think>reasoning here</think>\nActual response"
        result = re.sub(r"<think>.*?</think>\s*", "", msg, flags=re.DOTALL)
        self.assertEqual(result, "Actual response")

    def test_strip_multiline_think_tag(self):
        import re
        msg = "<think>\nLine 1\nLine 2\n</think>\n\nActual response"
        result = re.sub(r"<think>.*?</think>\s*", "", msg, flags=re.DOTALL)
        self.assertEqual(result, "Actual response")

    def test_no_think_tag_unchanged(self):
        import re
        msg = "Just a normal response"
        result = msg
        if "<think>" in msg:
            result = re.sub(r"<think>.*?</think>\s*", "", msg, flags=re.DOTALL)
        self.assertEqual(result, "Just a normal response")


if __name__ == "__main__":
    unittest.main()
