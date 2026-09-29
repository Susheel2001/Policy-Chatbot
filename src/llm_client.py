from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

try:
    from groq import Groq as GroqSDK
except ImportError:  # pragma: no cover - optional dependency until installed
    GroqSDK = None


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / '.env'
LEGACY_ENV_PATH = ROOT / '.env.example'
GROQ_DEFAULT_BASE_URL = 'https://api.groq.com/openai/v1'
GROQ_SDK_BASE_URL = 'https://api.groq.com'
GROQ_DEFAULT_MODEL = 'qwen/qwen3.8-27b'
GROQ_ENV_NAMES = ('GROQ_API_KEY', 'GROQ_BASE_URL', 'GROQ_MODEL')


@dataclass
class LLMConfig:
    api_key: str | None
    base_url: str | None
    model: str | None
    timeout: int = 60


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.api_key and self.config.base_url and self.config.model)

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1, max_tokens: int = 600) -> str:
        if not self.enabled:
            raise RuntimeError('LLM client is not configured.')

        if self._should_use_groq_sdk():
            return self._chat_with_groq_sdk(messages=messages, temperature=temperature, max_tokens=max_tokens)
        return self._chat_with_http(messages=messages, temperature=temperature, max_tokens=max_tokens)

    def _should_use_groq_sdk(self) -> bool:
        return bool(
            GroqSDK is not None
            and self.config.api_key
            and self.config.model
            and (
                not self.config.base_url
                or self.config.base_url.rstrip('/') == GROQ_DEFAULT_BASE_URL
            )
        )

    def _chat_with_groq_sdk(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        # The Groq SDK appends ``/openai/v1`` to its configured base URL.  The
        # app's OpenAI-compatible configuration, however, uses the complete
        # endpoint URL.  Passing that URL through unchanged would make the SDK
        # request ``/openai/v1/openai/v1/chat/completions``.
        client = GroqSDK(
            api_key=self.config.api_key,
            base_url=groq_sdk_base_url(self.config.base_url),
        )
        completion = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stream=True,
            stop=None,
        )
        parts: list[str] = []
        for chunk in completion:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                parts.append(delta)
        return ''.join(parts)

    def _chat_with_http(self, messages: list[dict[str, str]], temperature: float, max_tokens: int) -> str:
        url = self.config.base_url.rstrip('/') + '/chat/completions'
        payload = {
            'model': self.config.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        data = json.dumps(payload).encode('utf-8')
        req = request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')
        req.add_header('Authorization', f'Bearer {self.config.api_key}')

        with request.urlopen(req, timeout=self.config.timeout) as resp:
            body = resp.read().decode('utf-8')
        parsed = json.loads(body)
        return parsed['choices'][0]['message']['content']


def groq_sdk_base_url(base_url: str | None) -> str:
    """Convert an OpenAI-compatible Groq URL to the URL expected by its SDK."""
    normalized = (base_url or GROQ_SDK_BASE_URL).rstrip('/')
    return GROQ_SDK_BASE_URL if normalized == GROQ_DEFAULT_BASE_URL else normalized


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        if '=' not in line:
            continue

        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] in {'"', "'"} and value[-1:] == value[0]:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_env_files() -> None:
    load_env_file(ENV_PATH)
    if not any(os.getenv(name) for name in (*GROQ_ENV_NAMES, 'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL', 'POLICY_LLM_API_KEY', 'POLICY_LLM_BASE_URL', 'POLICY_LLM_MODEL')):
        load_env_file(LEGACY_ENV_PATH)


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_llm_client_from_env() -> OpenAICompatibleClient:
    load_env_files()

    groq_configured = any(os.getenv(name) for name in GROQ_ENV_NAMES)
    api_key = _first_env('GROQ_API_KEY', 'OPENAI_API_KEY', 'POLICY_LLM_API_KEY')
    base_url = _first_env('GROQ_BASE_URL', 'OPENAI_BASE_URL', 'POLICY_LLM_BASE_URL')
    model = _first_env('GROQ_MODEL', 'OPENAI_MODEL', 'POLICY_LLM_MODEL')

    if groq_configured:
        base_url = base_url or GROQ_DEFAULT_BASE_URL
        model = model or GROQ_DEFAULT_MODEL

    return OpenAICompatibleClient(
        LLMConfig(
            api_key=api_key,
            base_url=base_url or 'https://api.openai.com/v1',
            model=model,
        )
    )
