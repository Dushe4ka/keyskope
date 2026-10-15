# providers.py
"""Каталог LLM-провайдеров для KeyScope.

Ключ проверяется АВТОРИЗАЦИОННЫМ запросом — таким, который без валидного
ключа возвращает 401/403. Это важно: у части провайдеров (OpenRouter,
NVIDIA, SambaNova, AI21, Novita, ModelScope, Hugging Face) endpoint списка
моделей публичный и отдаёт 200 кому угодно, поэтому по нему нельзя
определять провайдера.

Способы пробы (probe_kind):
  chat        — POST chat/completions с несуществующей моделью. Авторизация
                проверяется раньше модели: валидный ключ -> 400/404 (модель
                не найдена, но ключ принят), невалидный -> 401/403.
  get_models  — GET списка моделей (у этих провайдеров он авторизованный).
  anthropic   — GET /v1/models с заголовками Anthropic.
  google      — GET /v1beta/models с x-goog-api-key (особый разбор ошибок).
  openrouter  — GET /api/v1/key (он же отдаёт баланс).
  hf_whoami   — GET huggingface.co/api/whoami-v2.

Баланс:
  balance_url + balance_kind — где провайдер отдаёт точную сумму через API
                               (OpenRouter, DeepSeek).
  У остальных точного API баланса нет. Для них наличие средств проверяется
  реальным мини-запросом генерации (max_tokens=1) живой моделью из models_url
  (см. check_funds в main.py) — api_style/models_url/gen_url это обслуживают.
"""
from dataclasses import dataclass, field
from typing import Optional

# Аутентификация.
AUTH_BEARER = "bearer"
AUTH_ANTHROPIC = "anthropic"
AUTH_GOOGLE = "google"

# Виды авторизационной пробы.
PROBE_CHAT = "chat"
PROBE_GET_MODELS = "get_models"
PROBE_ANTHROPIC = "anthropic"
PROBE_GOOGLE = "google"
PROBE_OPENROUTER = "openrouter"
PROBE_HF = "hf_whoami"

# Стиль API для запроса генерации (проверка наличия средств).
STYLE_OPENAI = "openai"
STYLE_ANTHROPIC = "anthropic"
STYLE_GOOGLE = "google"
STYLE_COHERE = "cohere"

# Чтение точного баланса.
BALANCE_OPENROUTER = "openrouter"
BALANCE_DEEPSEEK = "deepseek"

DUMMY_MODEL = "__keyscope_probe__"


@dataclass(frozen=True)
class Provider:
    id: str
    name: str
    probe_url: str
    probe_kind: str = PROBE_CHAT
    auth: str = AUTH_BEARER
    probe_model: str = DUMMY_MODEL
    # Для проверки наличия средств реальным запросом:
    api_style: str = STYLE_OPENAI
    models_url: Optional[str] = None      # GET список моделей ключа
    gen_url: Optional[str] = None         # endpoint генерации
    # Точный баланс через API (если есть):
    balance_url: Optional[str] = None
    balance_kind: Optional[str] = None
    prefixes: tuple = field(default_factory=tuple)


def _openai(id_, name, base, **kw):
    """OpenAI-совместимый: авторизация ловится POST {base}/chat/completions."""
    return Provider(id_, name, f"{base}/chat/completions", PROBE_CHAT,
                    models_url=f"{base}/models", gen_url=f"{base}/chat/completions", **kw)


def _openai_models(id_, name, base, **kw):
    """OpenAI-совместимый, но авторизацию ловит GET /models (chat проверяет модель раньше ключа)."""
    kw.setdefault("gen_url", f"{base}/chat/completions")
    return Provider(id_, name, f"{base}/models", PROBE_GET_MODELS,
                    models_url=f"{base}/models", **kw)


PROVIDERS = [
    _openai("openai", "OpenAI", "https://api.openai.com/v1", prefixes=("sk-", "sk-proj-")),
    Provider("anthropic", "Anthropic", "https://api.anthropic.com/v1/models",
             PROBE_ANTHROPIC, auth=AUTH_ANTHROPIC, api_style=STYLE_ANTHROPIC,
             models_url="https://api.anthropic.com/v1/models",
             gen_url="https://api.anthropic.com/v1/messages", prefixes=("sk-ant-",)),
    Provider("google", "Google Gemini",
             "https://generativelanguage.googleapis.com/v1beta/models",
             PROBE_GOOGLE, auth=AUTH_GOOGLE, api_style=STYLE_GOOGLE,
             models_url="https://generativelanguage.googleapis.com/v1beta/models",
             gen_url="https://generativelanguage.googleapis.com/v1beta",
             prefixes=("AIza", "AQ.")),
    Provider("openrouter", "OpenRouter", "https://openrouter.ai/api/v1/key",
             PROBE_OPENROUTER, balance_url="https://openrouter.ai/api/v1/key",
             balance_kind=BALANCE_OPENROUTER, prefixes=("sk-or-",)),
    _openai("deepseek", "DeepSeek", "https://api.deepseek.com",
            balance_url="https://api.deepseek.com/user/balance",
            balance_kind=BALANCE_DEEPSEEK, prefixes=("sk-",)),
    _openai("groq", "Groq", "https://api.groq.com/openai/v1", prefixes=("gsk_",)),
    _openai("mistral", "Mistral AI", "https://api.mistral.ai/v1"),
    _openai("together", "Together AI", "https://api.together.xyz/v1"),
    _openai_models("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1",
                   prefixes=("fw_",)),
    _openai_models("cohere", "Cohere", "https://api.cohere.com/v2",
                   api_style=STYLE_COHERE, gen_url="https://api.cohere.com/v2/chat"),
    _openai("xai", "xAI", "https://api.x.ai/v1", prefixes=("xai-",)),
    _openai("cerebras", "Cerebras", "https://api.cerebras.ai/v1", prefixes=("csk-",)),
    _openai("sambanova", "SambaNova", "https://api.sambanova.ai/v1",
            probe_model="Meta-Llama-3.1-8B-Instruct"),
    _openai("perplexity", "Perplexity", "https://api.perplexity.ai", prefixes=("pplx-",)),
    _openai("ai21", "AI21", "https://api.ai21.com/studio/v1"),
    _openai("nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1",
            probe_model="01-ai/yi-large", prefixes=("nvapi-",)),
    _openai("nebius", "Nebius AI Studio", "https://api.studio.nebius.ai/v1"),
    _openai("siliconflow", "SiliconFlow", "https://api.siliconflow.com/v1"),
    _openai_models("hyperbolic", "Hyperbolic", "https://api.hyperbolic.xyz/v1"),
    _openai("novita", "Novita AI", "https://api.novita.ai/v3/openai"),
    _openai("nscale", "Nscale", "https://inference.api.nscale.com/v1"),
    _openai("alibaba", "Alibaba Cloud / Qwen",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    _openai("modelscope", "ModelScope", "https://api-inference.modelscope.cn/v1",
            probe_model="Qwen/Qwen2.5-7B-Instruct", prefixes=("ms-",)),
    _openai("deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai"),
    Provider("huggingface", "Hugging Face", "https://huggingface.co/api/whoami-v2",
             PROBE_HF, models_url="https://router.huggingface.co/v1/models",
             gen_url="https://router.huggingface.co/v1/chat/completions",
             prefixes=("hf_",)),
]

PROVIDER_BY_ID = {provider.id: provider for provider in PROVIDERS}


def candidate_order(key: str) -> list:
    """Провайдеры с совпавшим префиксом — первыми (ускоряет auto-режим).
    Перебор всё равно охватывает весь каталог: структура ключа ничего не исключает.
    """
    preferred = [p for p in PROVIDERS if any(key.startswith(pre) for pre in p.prefixes)]
    rest = [p for p in PROVIDERS if p not in preferred]
    return preferred + rest
