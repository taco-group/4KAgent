"""Integration tests for MiniMax LLM provider.

These tests verify end-to-end integration with the MiniMax API.
They require MINIMAX_API_KEY to be set and will be skipped otherwise.
"""

import os
import json
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

import yaml


MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
SKIP_REASON = "MINIMAX_API_KEY not set"


def _make_config_with_key():
    return {
        "MINIMAX": {
            "API_KEY": MINIMAX_API_KEY,
            "MODEL": "MiniMax-M2.7",
            "MAX_TOKENS": 256,
            "TEMPERATURE": 0.7,
        }
    }


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxLLMIntegration(unittest.TestCase):
    """Live integration tests against MiniMax API."""

    def _make_llm(self, **kwargs):
        cfg = _make_config_with_key()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True, **kwargs)

    def test_simple_query(self):
        llm = self._make_llm()
        prompt, rsp = llm.query(prompt="Reply with exactly: PONG")
        self.assertIn("PONG", rsp)
        self.assertGreater(llm.prompt_tokens, 0)
        self.assertGreater(llm.completion_tokens, 0)

    def test_json_format_check(self):
        llm = self._make_llm()

        def check(obj):
            assert isinstance(obj, dict), "Response should be a dict"
            assert "answer" in obj, "Missing 'answer' key"

        prompt, rsp = llm.query(
            prompt='Return a JSON object with key "answer" and value "42". '
                   'Output ONLY the JSON, no markdown.',
            format_check=check,
        )
        parsed = eval(rsp)
        self.assertIn("answer", parsed)

    def test_system_message(self):
        llm = self._make_llm(system_message="Always respond in uppercase only.")
        _, rsp = llm.query(prompt="Say hello")
        upper_count = sum(1 for c in rsp if c.isupper())
        lower_count = sum(1 for c in rsp if c.islower())
        self.assertGreater(upper_count, lower_count)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestMiniMaxModelVariants(unittest.TestCase):
    """Test different MiniMax model variants."""

    def _make_llm(self, model):
        cfg = _make_config_with_key()
        cfg["MINIMAX"]["MODEL"] = model
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True)

    def test_m27_model(self):
        llm = self._make_llm("MiniMax-M2.7")
        _, rsp = llm.query(prompt="Reply OK")
        self.assertTrue(len(rsp) > 0)

    def test_m27_highspeed_model(self):
        llm = self._make_llm("MiniMax-M2.7-highspeed")
        _, rsp = llm.query(prompt="Reply OK")
        self.assertTrue(len(rsp) > 0)


@unittest.skipUnless(MINIMAX_API_KEY, SKIP_REASON)
class TestPipelineIntegration(unittest.TestCase):
    """Test that The4KAgent pipeline accepts minimax as llm_provider."""

    def test_llm_provider_arg_accepted(self):
        """Verify The4KAgent __init__ signature accepts llm_provider parameter."""
        import ast
        source_path = Path(__file__).parent.parent / "pipeline" / "the4kagent_pipeline.py"
        with open(source_path) as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                arg_names = [arg.arg for arg in node.args.args]
                if "llm_provider" in arg_names:
                    return
        self.fail("llm_provider not found in The4KAgent.__init__ signature")


if __name__ == "__main__":
    unittest.main()
