import logging
import threading
from typing import Optional
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = 'openai'


logger = logging.getLogger(__name__)


class OpenAIClient(BaseClient):
    ClientClass = OpenAI

    MODEL_CAPS = {
        "gpt-5": {"supports_native_n": True, "max_native_n": 8},
        "gpt-5-mini": {"supports_native_n": True, "max_native_n": 8},

        # 举例：很多 reasoning model 你可以先保守设成不支持
        "o3-mini": {"supports_native_n": False, "max_native_n": 1},
        "deepseek-r1": {"supports_native_n": False, "max_native_n": 1},
        "kimi-k2-thinking": {"supports_native_n": False, "max_native_n": 1},
    }

    DEFAULT_SUPPORTS_NATIVE_N = True
    DEFAULT_MAX_NATIVE_N = None

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 100,
    ) -> None:
        super().__init__(model, temperature)

        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)

        self.client = self.client = self.ClientClass(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
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
            temperature=1.0
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                n=n,
                stream=False,
            )

        except Exception as e:
            err_msg = str(e).lower()

            # temperature 不被支持 / 超上限
            if "temperature" in err_msg:
                logger.warning(
                    f"Model {self.model} rejected temperature={temperature}; "
                    f"retrying with temperature=1."
                )

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1,
                    n=n,
                    stream=False,
                )
                self._force_temperature_one = True
            else:
                raise

        self._record_usage(response)
        return response.choices

    def _emulate_n_by_repeated_calls(self, messages: list[dict], temperature: float, n: int):
        logger.info(
            f"Model {self.model} does not support native n>1; "
            f"emulating n={n} with repeated n=1 calls."
        )
        all_choices = []
        for _ in range(n):
            all_choices.extend(self._single_chat_completion_api(messages, temperature, n=1))
        return all_choices

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        caps = self._get_model_caps()
        supports_native_n = caps["supports_native_n"]
        max_native_n = caps["max_native_n"]

        # 完全不支持 n>1
        if not supports_native_n:
            if n == 1:
                return self._single_chat_completion_api(messages, temperature, n=1)
            return self._emulate_n_by_repeated_calls(messages, temperature, n)

        # 支持 native n，且无上限配置
        if max_native_n is None or n <= max_native_n:
            try:
                return self._single_chat_completion_api(messages, temperature, n)
            except Exception as e:
                err_msg = str(e).lower()

                # 自动降级：接口虽然没配置，但其实不支持 n>1
                if n > 1 and (
                    "param: 'n'" in err_msg
                    or "invalid 'n'" in err_msg
                    or "unsupported" in err_msg and "'n'" in err_msg
                ):
                    logger.warning(
                        f"Model {self.model} rejected native n={n}; "
                        f"falling back to repeated n=1 calls."
                    )
                    return self._emulate_n_by_repeated_calls(messages, temperature, n)
                raise

        # 支持 native n，但有上限，自动分批
        logger.info(
            f"Model {self.model} only supports n <= {max_native_n}. "
            f"Requested n={n}; automatically splitting into batches."
        )
        all_choices = []
        remaining = n
        while remaining > 0:
            cur_n = min(remaining, max_native_n)
            all_choices.extend(self._single_chat_completion_api(messages, temperature, cur_n))
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
            
            
    def chat_completion_with_tools(self,
        messages: list[dict],
        tools: list[dict],
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
    ):
        temperature = self.temperature if temperature is None else temperature

        if self._force_temperature_one:
            temperature = 1.0

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=temperature,
                stream=False,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "temperature" in err_msg:
                logger.warning(
                    f"Model {self.model} rejected temperature={temperature}; retrying with temperature=1."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    temperature=1,
                    stream=False,
                )
                self._force_temperature_one = True
            else:
                raise

        self._record_usage(response)
        return response
    
    
