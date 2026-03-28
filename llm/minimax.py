from pathlib import Path
import requests
import logging
from typing import Callable, Optional
from time import sleep
import random
import re

from .base_llm import BaseLLM
from utils.misc import encode_img


class MiniMaxLLM(BaseLLM):
    """MiniMax LLM provider using the OpenAI-compatible API.

    Parameters when called: img_path_lst, prompt, format_check.
    """

    MINIMAX_API_URL = "https://api.minimax.io/v1/chat/completions"

    def __init__(self,
                 config_path: Path = Path("config.yml"),
                 log_path: Optional[Path | str] = None,
                 logger: Optional[logging.Logger] = None,
                 silent: bool = False,
                 system_message: Optional[str] = None,
                 model: Optional[str] = None
                 ):
        super().__init__(
            config_path=config_path,
            log_path=log_path,
            logger=logger,
            silent=silent
        )

        self.api_key = self.cfg["MINIMAX"]["API_KEY"]
        if model is None:
            self.model = self.cfg["MINIMAX"]["MODEL"]
        else:
            self.model = model
        self.max_tokens = self.cfg["MINIMAX"]["MAX_TOKENS"]
        # MiniMax requires temperature in (0.0, 1.0]
        raw_temp = self.cfg["MINIMAX"]["TEMPERATURE"]
        self.temperature = max(0.01, min(float(raw_temp), 1.0))

        self.prompt_tokens = 0
        self.completion_tokens = 0

        self.system_message = system_message
        if self.system_message is not None:
            self._log("_Note: These user-assistant interactions are independent "
                      "and the system message is always attached in each turn for MiniMax._")
            self._log("**System message for MiniMax**")
            self._log(self.system_message)

    def query(self,
              img_path_lst: Optional[list[Path]] = None,
              prompt: str = "",
              format_check: Optional[Callable[[object], None]] = None,
              ) -> tuple[str, str]:
        headers, payload = self._prepare_for_request(
            prompt, img_path_lst)
        while True:
            response = self._send_request(headers, payload)

            usage = response.json()["usage"]
            self.prompt_tokens += usage["prompt_tokens"]
            self.completion_tokens += usage["completion_tokens"]

            rsp_text: str = response.json()['choices'][0]['message']['content']
            # Strip MiniMax thinking tags if present
            rsp_text = re.sub(r'<think>.*?</think>\s*', '', rsp_text, flags=re.DOTALL).strip()
            if format_check is not None:
                valid, rsp_text = self._check_syntax(rsp_text, format_check)
                if not valid:
                    continue
            return prompt, rsp_text

    def _prepare_for_request(self, prompt: str,
                             img_path_lst: Optional[list[Path]] = None
                             ) -> tuple[dict, dict]:
        content = [{
            "type": "text",
            "text": prompt
        }]
        if img_path_lst is not None:
            for img_path in img_path_lst:
                img_base64 = encode_img(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64,
                        "detail": "auto"
                    }
                })

        messages = []
        if self.system_message is not None:
            messages.append({
                "role": "system",
                "content": self.system_message
            })
        messages.append({
            "role": "user",
            "content": content
        })

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }

        return headers, payload

    def _send_request(self, headers: dict, payload: dict,
                      max_retries: int = 5,
                      initial_delay: int = 3,
                      exp_base: int = 2,
                      jitter: bool = True) -> requests.Response:
        """Sends a request to the MiniMax API and handles errors with exponential backoff."""

        n_retries = 0
        backoff_delay = initial_delay
        while True:
            try:
                response = requests.post(self.MINIMAX_API_URL,
                                         headers=headers, json=payload)
                is_valid, recommended_delay = self._check_response(response)
                if is_valid:
                    return response
            except Exception as e:
                self._log("An error occurred when sending a request: "
                          f"{type(e).__name__}: {e}",
                          level='warning')
                recommended_delay = None

            n_retries += 1
            if n_retries > max_retries:
                raise RuntimeError(
                    "Too many errors occurred when querying MiniMax LLM.")
            if recommended_delay is not None:
                delay = recommended_delay
            else:
                backoff_delay *= exp_base * (1 + jitter*random.random())
                delay = backoff_delay
            self._log(
                f"Retrying in {delay:.3f} seconds...", level='warning')
            sleep(delay)

    def _check_response(self, response: requests.Response) -> tuple[bool, Optional[float]]:
        """Checks if the response is valid. If error occurs, gets the recommended delay if any."""

        if "error" in response.json():
            err_msg: str = response.json()['error']['message']
            self._log(f"An error occurred when querying MiniMax LLM: {err_msg}",
                      level='warning')

            recommended_delay = None
            if response.json()['error'].get('code') == 'rate_limit_exceeded':
                match = re.search(
                    R"(?<=Please try again in )(\d+m)?\d+\.?\d*(?=s)", err_msg)
                if match is not None:
                    t = match.group().split('m')
                    m = t[0] if len(t) > 1 else 0
                    s = t[-1]
                    recommended_delay = 60*int(m) + float(s)

            return False, recommended_delay

        if (finish_reason := response.json()['choices'][0]['finish_reason']) != 'stop':
            self._log(f"finish_reason is {finish_reason}", level='warning')

        return True, None

    def _check_syntax(self, rsp_text: str, format_check: Callable[[object], None]
                      ) -> tuple[bool, str]:
        """Checks whether the response is a valid Python object and follows the specified format."""
        try:
            obj = eval(rsp_text)
        except:
            inner_rsp_text = rsp_text.strip("```").lstrip("json").strip()
            try:
                obj = eval(inner_rsp_text)
                rsp_text = inner_rsp_text
            except:
                self._log("Failed to parse the response:", level='warning')
                self._log(rsp_text, level='warning')
                return False, ""
        try:
            format_check(obj)
        except AssertionError as e:
            self._log(f"Failed to pass the format check: {e}", level='warning')
            self._log(f"Response: {obj}", level='warning')
            return False, ""
        return True, rsp_text

    def _post_process(self):
        """Logs the token usage."""
        self._log("Token usage so far: "
                  f"{self.prompt_tokens} prompt tokens, "
                  f"{self.completion_tokens} completion tokens")
