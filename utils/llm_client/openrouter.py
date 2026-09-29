import logging
import threading
from typing import Optional
from omegaconf import OmegaConf
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = "openai"

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseClient):

    ClientClass = OpenAI

    MODEL_CAPS = {
        "gpt-5": {"supports_native_n": True, "max_native_n": 8},
        "gpt-5-mini": {"supports_native_n": True, "max_native_n": 8},

        "o3-mini": {"supports_native_n": False, "max_native_n": 1},
        "deepseek-r1": {"supports_native_n": False, "max_native_n": 1},
    }

    DEFAULT_SUPPORTS_NATIVE_N = True
    DEFAULT_MAX_NATIVE_N = None

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 1000,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        reasoning: Optional[dict] = None,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            raise ImportError("Please install the `openai` package.")

        self.client = self.ClientClass(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

        extra_body_list = ["thinking", "step-3.5"]

        self.extra_headers = {}
        if site_url:
            self.extra_headers["HTTP-Referer"] = site_url
        if app_name:
            self.extra_headers["X-Title"] = app_name

        if reasoning and reasoning.get("enabled") and any(x in model for x in extra_body_list):
            self.reasoning = OmegaConf.to_container(reasoning, resolve=True)
        else:
            self.reasoning = None

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_calls = 0

        self._usage_lock = threading.Lock()
        self._force_temperature_one = False

    def _get_model_caps(self):
        model_lower = self.model.lower()
        for prefix, caps in self.MODEL_CAPS.items():
            if model_lower.startswith(prefix):
                return caps
        return {
            "supports_native_n": self.DEFAULT_SUPPORTS_NATIVE_N,
            "max_native_n": self.DEFAULT_MAX_NATIVE_N,
        }

    def _record_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is not None:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            with self._usage_lock:
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                self.total_calls += 1
        else:
            with self._usage_lock:
                self.total_calls += 1

    def _single_chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):

        if self._force_temperature_one:
            temperature = 1.0

        create_kwargs = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            n=n,
            stream=False,
            extra_headers=self.extra_headers if self.extra_headers else None,
        )

        if self.reasoning is not None:
            create_kwargs["extra_body"] = {"reasoning": self.reasoning}

        try:
            response = self.client.chat.completions.create(**create_kwargs)

        except Exception as e:
            err_msg = str(e).lower()

            if "temperature" in err_msg:
                logger.warning(
                    f"Model {self.model} rejected temperature={temperature}; retrying with temperature=1."
                )

                create_kwargs["temperature"] = 1
                response = self.client.chat.completions.create(**create_kwargs)
                self._force_temperature_one = True
            else:
                raise

        self._record_usage(response)
        return response.choices

    def _emulate_n_by_repeated_calls(self, messages: list[dict], temperature: float, n: int):
        logger.info(
            f"Model {self.model} does not support native n>1; emulating n={n} with repeated n=1 calls."
        )

        all_choices = []
        for _ in range(n):
            all_choices.extend(self._single_chat_completion_api(messages, temperature, n=1))

        return all_choices

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):

        caps = self._get_model_caps()
        supports_native_n = caps["supports_native_n"]
        max_native_n = caps["max_native_n"]

        if not supports_native_n:
            if n == 1:
                return self._single_chat_completion_api(messages, temperature, n=1)
            return self._emulate_n_by_repeated_calls(messages, temperature, n)

        if max_native_n is None or n <= max_native_n:
            try:
                return self._single_chat_completion_api(messages, temperature, n)
            except Exception as e:
                err_msg = str(e).lower()

                if n > 1 and (
                    "param: 'n'" in err_msg
                    or "invalid 'n'" in err_msg
                    or ("unsupported" in err_msg and "'n'" in err_msg)
                ):
                    logger.warning(
                        f"Model {self.model} rejected native n={n}; falling back to repeated n=1 calls."
                    )
                    return self._emulate_n_by_repeated_calls(messages, temperature, n)

                raise

        logger.info(
            f"Model {self.model} only supports n <= {max_native_n}. Requested n={n}; automatically splitting into batches."
        )

        all_choices = []
        remaining = n

        while remaining > 0:
            cur_n = min(remaining, max_native_n)
            all_choices.extend(
                self._single_chat_completion_api(messages, temperature, cur_n)
            )
            remaining -= cur_n

        return all_choices

    def get_usage_summary(self) -> dict:
        with self._usage_lock:
            return {
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
                "total_calls": self.total_calls,
            }

    def reset_usage_summary(self) -> None:
        with self._usage_lock:
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
            self.total_calls = 0
            
    def chat_completion_with_tools(self,
    messages: list[dict],
    tools: list[dict],
    tool_choice: str = "auto",
    temperature: Optional[float] = None,
):
        if temperature is None:
            temperature = self.temperature

        if self._force_temperature_one:
            temperature = 1.0

        create_kwargs = dict(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            stream=False,
            extra_headers=self.extra_headers if self.extra_headers else None,
        )

        if self.reasoning is not None:
            create_kwargs["extra_body"] = {"reasoning": self.reasoning}

        try:
            response = self.client.chat.completions.create(**create_kwargs)

        except Exception as e:
            err_msg = str(e).lower()

            if "temperature" in err_msg:
                logger.warning(
                    f"Model {self.model} rejected temperature={temperature}; retrying with temperature=1."
                )

                create_kwargs["temperature"] = 1
                response = self.client.chat.completions.create(**create_kwargs)
                self._force_temperature_one = True
            else:
                raise

        self._record_usage(response)
        return response