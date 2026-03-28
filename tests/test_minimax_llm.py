"""Unit tests for llm/minimax.py — MiniMaxLLM provider."""

import json
import sys
import os
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import yaml

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(overrides=None):
    """Return a minimal config dict that MiniMaxLLM can consume."""
    cfg = {
        "MINIMAX": {
            "API_KEY": "test-key-123",
            "MODEL": "MiniMax-M2.7",
            "MAX_TOKENS": 3000,
            "TEMPERATURE": 0.7,
        }
    }
    if overrides:
        cfg["MINIMAX"].update(overrides)
    return cfg


def _make_api_response(content="hello", prompt_tokens=10, completion_tokens=5,
                       finish_reason="stop", error=None):
    """Build a mock requests.Response from the MiniMax API."""
    resp = MagicMock()
    if error:
        body = {"error": error}
    else:
        body = {
            "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    resp.json.return_value = body
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMiniMaxLLMInit(unittest.TestCase):
    """Test MiniMaxLLM initialization and configuration."""

    def _make_llm(self, cfg=None, **kwargs):
        cfg = cfg or _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), **kwargs)

    def test_default_model_from_config(self):
        llm = self._make_llm()
        self.assertEqual(llm.model, "MiniMax-M2.7")

    def test_model_override(self):
        llm = self._make_llm(model="MiniMax-M2.7-highspeed")
        self.assertEqual(llm.model, "MiniMax-M2.7-highspeed")

    def test_api_key_loaded(self):
        llm = self._make_llm()
        self.assertEqual(llm.api_key, "test-key-123")

    def test_max_tokens(self):
        llm = self._make_llm()
        self.assertEqual(llm.max_tokens, 3000)

    def test_temperature_clamp_zero(self):
        """Temperature 0.0 should be clamped to 0.01."""
        cfg = _make_config({"TEMPERATURE": 0.0})
        llm = self._make_llm(cfg)
        self.assertAlmostEqual(llm.temperature, 0.01)

    def test_temperature_clamp_above_one(self):
        """Temperature > 1.0 should be clamped to 1.0."""
        cfg = _make_config({"TEMPERATURE": 1.5})
        llm = self._make_llm(cfg)
        self.assertAlmostEqual(llm.temperature, 1.0)

    def test_temperature_normal_range(self):
        cfg = _make_config({"TEMPERATURE": 0.7})
        llm = self._make_llm(cfg)
        self.assertAlmostEqual(llm.temperature, 0.7)

    def test_initial_token_counters(self):
        llm = self._make_llm()
        self.assertEqual(llm.prompt_tokens, 0)
        self.assertEqual(llm.completion_tokens, 0)

    def test_system_message_stored(self):
        llm = self._make_llm(system_message="You are helpful.")
        self.assertEqual(llm.system_message, "You are helpful.")

    def test_system_message_none_by_default(self):
        llm = self._make_llm()
        self.assertIsNone(llm.system_message)


class TestMiniMaxLLMPrepareRequest(unittest.TestCase):
    """Test request payload construction."""

    def _make_llm(self, **kwargs):
        cfg = _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), **kwargs)

    def test_text_only_payload(self):
        llm = self._make_llm()
        headers, payload = llm._prepare_for_request("hello world")
        self.assertEqual(headers["Authorization"], "Bearer test-key-123")
        self.assertEqual(payload["model"], "MiniMax-M2.7")
        self.assertEqual(len(payload["messages"]), 1)
        self.assertEqual(payload["messages"][0]["content"][0]["text"], "hello world")

    def test_system_message_prepended(self):
        llm = self._make_llm(system_message="system msg")
        _, payload = llm._prepare_for_request("user msg")
        self.assertEqual(len(payload["messages"]), 2)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][0]["content"], "system msg")

    def test_image_payload(self):
        llm = self._make_llm()
        with patch("llm.minimax.encode_img", return_value="data:image/png;base64,abc"):
            _, payload = llm._prepare_for_request("describe", [Path("img.png")])
        user_content = payload["messages"][0]["content"]
        self.assertEqual(len(user_content), 2)
        self.assertEqual(user_content[1]["type"], "image_url")

    def test_payload_includes_temperature(self):
        llm = self._make_llm()
        _, payload = llm._prepare_for_request("test")
        self.assertAlmostEqual(payload["temperature"], 0.7)

    def test_payload_includes_max_tokens(self):
        llm = self._make_llm()
        _, payload = llm._prepare_for_request("test")
        self.assertEqual(payload["max_tokens"], 3000)


class TestMiniMaxLLMCheckResponse(unittest.TestCase):
    """Test response validation."""

    def _make_llm(self):
        cfg = _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True)

    def test_valid_response(self):
        llm = self._make_llm()
        resp = _make_api_response("ok")
        is_valid, delay = llm._check_response(resp)
        self.assertTrue(is_valid)
        self.assertIsNone(delay)

    def test_error_response(self):
        llm = self._make_llm()
        resp = _make_api_response(error={"message": "bad request", "code": "invalid_request"})
        is_valid, delay = llm._check_response(resp)
        self.assertFalse(is_valid)

    def test_rate_limit_with_retry_hint(self):
        llm = self._make_llm()
        resp = _make_api_response(error={
            "message": "Please try again in 5s.",
            "code": "rate_limit_exceeded"
        })
        is_valid, delay = llm._check_response(resp)
        self.assertFalse(is_valid)
        self.assertAlmostEqual(delay, 5.0)

    def test_rate_limit_with_minute_hint(self):
        llm = self._make_llm()
        resp = _make_api_response(error={
            "message": "Please try again in 1m30s.",
            "code": "rate_limit_exceeded"
        })
        is_valid, delay = llm._check_response(resp)
        self.assertFalse(is_valid)
        self.assertAlmostEqual(delay, 90.0)

    def test_non_stop_finish_reason(self):
        llm = self._make_llm()
        resp = _make_api_response("partial", finish_reason="length")
        is_valid, _ = llm._check_response(resp)
        self.assertTrue(is_valid)


class TestMiniMaxLLMQuery(unittest.TestCase):
    """Test the query method including think-tag stripping."""

    def _make_llm(self, **kwargs):
        cfg = _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True, **kwargs)

    @patch("llm.minimax.requests.post")
    def test_basic_query(self, mock_post):
        mock_post.return_value = _make_api_response("answer text")
        llm = self._make_llm()
        prompt, rsp = llm.query(prompt="hello")
        self.assertEqual(prompt, "hello")
        self.assertEqual(rsp, "answer text")

    @patch("llm.minimax.requests.post")
    def test_thinking_tags_stripped(self, mock_post):
        content = "<think>internal reasoning</think>\nactual answer"
        mock_post.return_value = _make_api_response(content)
        llm = self._make_llm()
        _, rsp = llm.query(prompt="test")
        self.assertEqual(rsp, "actual answer")

    @patch("llm.minimax.requests.post")
    def test_thinking_tags_multiline_stripped(self, mock_post):
        content = "<think>\nstep 1\nstep 2\n</think>\n\nfinal result"
        mock_post.return_value = _make_api_response(content)
        llm = self._make_llm()
        _, rsp = llm.query(prompt="test")
        self.assertEqual(rsp, "final result")

    @patch("llm.minimax.requests.post")
    def test_token_counting(self, mock_post):
        mock_post.return_value = _make_api_response("ok", prompt_tokens=100, completion_tokens=50)
        llm = self._make_llm()
        llm.query(prompt="test")
        self.assertEqual(llm.prompt_tokens, 100)
        self.assertEqual(llm.completion_tokens, 50)

    @patch("llm.minimax.requests.post")
    def test_token_accumulation(self, mock_post):
        mock_post.return_value = _make_api_response("ok", prompt_tokens=10, completion_tokens=5)
        llm = self._make_llm()
        llm.query(prompt="q1")
        llm.query(prompt="q2")
        self.assertEqual(llm.prompt_tokens, 20)
        self.assertEqual(llm.completion_tokens, 10)

    @patch("llm.minimax.requests.post")
    def test_api_url(self, mock_post):
        mock_post.return_value = _make_api_response("ok")
        llm = self._make_llm()
        llm.query(prompt="test")
        url = mock_post.call_args[0][0]
        self.assertEqual(url, "https://api.minimax.io/v1/chat/completions")


class TestMiniMaxLLMCheckSyntax(unittest.TestCase):
    """Test response parsing and format checking."""

    def _make_llm(self):
        cfg = _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True)

    def test_valid_python_object(self):
        llm = self._make_llm()
        valid, text = llm._check_syntax('{"key": "value"}', lambda x: None)
        self.assertTrue(valid)

    def test_code_block_unwrapping(self):
        llm = self._make_llm()
        valid, text = llm._check_syntax('```json\n{"key": "value"}\n```', lambda x: None)
        self.assertTrue(valid)

    def test_format_check_failure(self):
        llm = self._make_llm()
        def strict_check(obj):
            assert "required" in obj, "Missing required key"
        valid, text = llm._check_syntax('{"key": "value"}', strict_check)
        self.assertFalse(valid)

    def test_invalid_syntax(self):
        llm = self._make_llm()
        valid, text = llm._check_syntax('not valid python {[}', lambda x: None)
        self.assertFalse(valid)


class TestMiniMaxLLMSendRequest(unittest.TestCase):
    """Test retry logic in _send_request."""

    def _make_llm(self):
        cfg = _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True)

    @patch("llm.minimax.sleep")
    @patch("llm.minimax.requests.post")
    def test_retry_on_error_then_success(self, mock_post, mock_sleep):
        error_resp = _make_api_response(error={"message": "server error", "code": "server_error"})
        ok_resp = _make_api_response("ok")
        mock_post.side_effect = [error_resp, ok_resp]
        llm = self._make_llm()
        result = llm._send_request({"Authorization": "Bearer k"}, {"model": "m"})
        self.assertEqual(result.json()["choices"][0]["message"]["content"], "ok")
        self.assertEqual(mock_post.call_count, 2)

    @patch("llm.minimax.sleep")
    @patch("llm.minimax.requests.post")
    def test_max_retries_exceeded(self, mock_post, mock_sleep):
        error_resp = _make_api_response(error={"message": "error", "code": "server_error"})
        mock_post.return_value = error_resp
        llm = self._make_llm()
        with self.assertRaises(RuntimeError):
            llm._send_request({"Authorization": "Bearer k"}, {"model": "m"}, max_retries=2)

    @patch("llm.minimax.sleep")
    @patch("llm.minimax.requests.post")
    def test_retry_on_exception(self, mock_post, mock_sleep):
        ok_resp = _make_api_response("ok")
        mock_post.side_effect = [ConnectionError("timeout"), ok_resp]
        llm = self._make_llm()
        result = llm._send_request({"Authorization": "Bearer k"}, {"model": "m"})
        self.assertEqual(result.json()["choices"][0]["message"]["content"], "ok")


class TestMiniMaxLLMPostProcess(unittest.TestCase):
    """Test post-processing (logging)."""

    def _make_llm(self):
        cfg = _make_config()
        with patch("builtins.open", mock_open(read_data=yaml.dump(cfg))):
            from llm.minimax import MiniMaxLLM
            return MiniMaxLLM(config_path=Path("dummy.yml"), silent=True)

    def test_post_process_does_not_raise(self):
        llm = self._make_llm()
        llm.prompt_tokens = 100
        llm.completion_tokens = 50
        llm._post_process()


class TestMiniMaxAPIURL(unittest.TestCase):
    """Ensure the API URL constant is correct."""

    def test_api_url_constant(self):
        from llm.minimax import MiniMaxLLM
        self.assertEqual(MiniMaxLLM.MINIMAX_API_URL, "https://api.minimax.io/v1/chat/completions")


class TestConfigYML(unittest.TestCase):
    """Test that config.yml includes MINIMAX section."""

    def test_minimax_section_exists(self):
        with open(Path(__file__).parent.parent / "config.yml") as f:
            cfg = yaml.safe_load(f)
        self.assertIn("MINIMAX", cfg)
        self.assertIn("API_KEY", cfg["MINIMAX"])
        self.assertIn("MODEL", cfg["MINIMAX"])
        self.assertEqual(cfg["MINIMAX"]["MODEL"], "MiniMax-M2.7")

    def test_minimax_temperature_range(self):
        with open(Path(__file__).parent.parent / "config.yml") as f:
            cfg = yaml.safe_load(f)
        temp = cfg["MINIMAX"]["TEMPERATURE"]
        self.assertGreater(temp, 0.0)
        self.assertLessEqual(temp, 1.0)


class TestModuleExports(unittest.TestCase):
    """Test that MiniMaxLLM is importable from llm package."""

    def test_import(self):
        from llm.minimax import MiniMaxLLM
        self.assertTrue(hasattr(MiniMaxLLM, 'query'))
        self.assertTrue(hasattr(MiniMaxLLM, '_prepare_for_request'))
        self.assertTrue(hasattr(MiniMaxLLM, '_send_request'))


if __name__ == "__main__":
    unittest.main()
