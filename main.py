# main.py
import asyncio
import threading
import time
import webbrowser
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from providers import (
    AUTH_ANTHROPIC,
    AUTH_GOOGLE,
    BALANCE_DEEPSEEK,
    BALANCE_OPENROUTER,
    PROBE_CHAT,
    PROBE_GOOGLE,
    PROVIDERS,
    PROVIDER_BY_ID,
    Provider,
    STYLE_ANTHROPIC,
    STYLE_COHERE,
    STYLE_GOOGLE,
    candidate_order,
)


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="KeyScope")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Короткие таймауты: проба — лёгкий GET, незачем ждать 30 секунд при переборе.
PROBE_TIMEOUT = httpx.Timeout(12.0, connect=8.0)
MAX_CONCURRENT_PROBES = 30
MAX_SCAN_KEYS = 200

# Браузероподобный User-Agent снижает число ложных блокировок от WAF/cloudflare.
CLIENT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (KeyScope; +local) httpx",
    "Accept": "application/json",
}


class CheckRequest(BaseModel):
    key: str = Field(min_length=4, max_length=2000)
    provider: str = "auto"


class ScanRequest(BaseModel):
    keys: list[str] = Field(min_length=1, max_length=MAX_SCAN_KEYS)


def mask_key(key: str) -> str:
    if len(key) <= 12:
        return "••••••••"
    return f"{key[:6]}••••••••{key[-4:]}"


def auth_headers(provider: Provider, key: str) -> dict:
    if provider.auth == AUTH_ANTHROPIC:
        return {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if provider.auth == AUTH_GOOGLE:
        # x-goog-api-key принимает и обычные AIza-ключи, и новые AQ.-ключи.
        return {"x-goog-api-key": key}
    return {"Authorization": f"Bearer {key}"}


def response_error(response: httpx.Response) -> str:
    """Короткое сообщение об ошибке из ответа API (без полного тела)."""
    try:
        data = response.json()
    except ValueError:
        return ""
    if not isinstance(data, dict):
        return ""
    error = data.get("error", data)
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail") or ""
        code = error.get("code") or error.get("type") or ""
        return f"{message} {code}".strip()[:300].lower()
    if isinstance(error, str):
        return error[:300].lower()
    return ""


QUOTA_TERMS = (
    "insufficient_quota", "billing", "credit balance", "out of credits",
    "quota exceeded", "payment required", "exceeded your current quota",
    "no credit", "balance is insufficient", "insufficient balance",
)

# Признаки того, что провайдер отверг именно КЛЮЧ (а не модель/формат).
# Нужны для статусов 400/404, где авторизация и ошибка запроса смешаны.
AUTH_FAIL_TERMS = (
    "api key not valid", "invalid api key", "incorrect api key",
    "invalid_api_key", "unauthorized", "authentication", "no auth",
    "invalid key", "api_key_invalid", "invalid token", "expired",
)

def _quota(text: str) -> bool:
    return any(term in text for term in QUOTA_TERMS)


def _auth_fail(text: str) -> bool:
    return any(term in text for term in AUTH_FAIL_TERMS)


def classify_auth(response: httpx.Response) -> tuple[str, str]:
    """Разбор ответа авторизационной пробы (chat / get_models / anthropic / openrouter)."""
    code = response.status_code
    text = response_error(response)

    if response.is_success:
        return "valid", "Ключ принят — активен."
    if code == 401:
        return "invalid", "Авторизация не пройдена: ключ неверный, отозван или недействителен."
    if code == 402:
        return "no_credits", "HTTP 402: требуется пополнение или оплата."
    if code == 403:
        if _quota(text):
            return "no_credits", "Доступ закрыт из-за исчерпанной квоты/оплаты."
        if _auth_fail(text):
            return "invalid", "Ключ отклонён."
        if not text:
            # 403 без JSON (часто cloudflare/WAF) — не подтверждает принадлежность ключа.
            return "blocked", "Доступ закрыт (WAF/прокси?) — провайдер не подтверждён."
        return "forbidden", "Доступ запрещён: ключ есть, но нет прав (ограничения/отключён)."
    if code == 429:
        if _quota(text):
            return "no_credits", "Исчерпана квота или проблема с оплатой."
        return "rate_limited", "HTTP 429: лимит запросов. Ключ активен."
    # 400/404/422: для chat-пробы это обычно «модель не найдена» при принятом ключе.
    if code in (400, 404, 422):
        if _auth_fail(text):
            return "invalid", "Ключ отклонён."
        if text:  # есть осмысленная JSON-ошибка -> ключ принят, отвергнута модель/запрос
            return "valid", "Ключ принят (тестовая модель не найдена — это ожидаемо)."
        # Пустой/не-JSON ответ (например, прокси или заглушка) — не считаем совпадением.
        return "request_error", "Непонятный ответ без деталей — провайдер не подтверждён."
    if code >= 500:
        return "provider_error", "Сервер провайдера вернул ошибку."
    return "error", f"Неожиданный HTTP-статус: {code}."


def classify_google(response: httpx.Response) -> tuple[str, str]:
    """У Google много разных ошибок на одном коде — разбираем по тексту."""
    if response.is_success:
        return "valid", "Ключ принят — активен."
    text = response_error(response)
    if "location is not supported" in text or "user location" in text:
        return "region_blocked", "Ключ валиден, но запрещён ваш регион (нужен VPN/другой регион)."
    if _auth_fail(text) or "api key not valid" in text:
        return "invalid", "Ключ неверный или недействителен."
    if "permission denied" in text or response.status_code == 403:
        return "forbidden", "Ключ есть, но доступ запрещён (ключ ограничен/отключён или API выключен)."
    if _quota(text):
        return "no_credits", "Исчерпана квота."
    if response.status_code == 429:
        return "rate_limited", "Лимит запросов. Ключ активен."
    return "invalid", "Ключ не принят."


def _chat_payload(provider: Provider) -> dict:
    return {"model": provider.probe_model,
            "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}


async def probe_provider(client: httpx.AsyncClient, provider: Provider, key: str) -> dict:
    """Авторизационная проба: подтверждает, что ключ принят именно этим провайдером."""
    headers = auth_headers(provider, key)
    try:
        if provider.probe_kind == PROBE_CHAT:
            response = await client.post(provider.probe_url, headers=headers,
                                         json=_chat_payload(provider))
        else:
            response = await client.get(provider.probe_url, headers=headers)
    except httpx.TimeoutException:
        return {"provider": provider.id, "provider_name": provider.name,
                "status": "timeout", "message": "Провайдер не ответил вовремя.",
                "status_code": None}
    except httpx.HTTPError:
        return {"provider": provider.id, "provider_name": provider.name,
                "status": "network_error",
                "message": "Не удалось подключиться к API провайдера.", "status_code": None}

    if provider.probe_kind == PROBE_GOOGLE:
        status, message = classify_google(response)
    else:
        status, message = classify_auth(response)

    result = {
        "provider": provider.id,
        "provider_name": provider.name,
        "status": status,
        "message": message,
        "status_code": response.status_code,
    }

    # Число моделей — бонус для GET-проб, вернувших список.
    if response.is_success and provider.probe_kind != PROBE_CHAT:
        try:
            data = response.json()
            items = data.get("data", data) if isinstance(data, dict) else data
            if isinstance(items, dict):
                items = items.get("models", items.get("data"))
            if isinstance(items, list):
                result["models_count"] = len(items)
        except ValueError:
            pass

    return result


def _num(value) -> float | None:
    """Безопасно приводит строку/число к float (DeepSeek отдаёт строки)."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --- Проверка наличия средств реальным мини-запросом (где нет API баланса) ---

# Сколько моделей из списка пробовать, прежде чем сдаться (на случай, если
# первая модель — не чат, требует доступа и т.п.).
MAX_MODEL_TRIES = 3


async def list_models(client: httpx.AsyncClient, provider: Provider, key: str) -> list:
    """Возвращает id моделей ключа (для пробного запроса). Пустой список — если не вышло."""
    if not provider.models_url:
        return []
    try:
        r = await client.get(provider.models_url, headers=auth_headers(provider, key))
        if not r.is_success:
            return []
        data = r.json()
    except (httpx.HTTPError, ValueError):
        return []

    ids = []
    if provider.api_style == STYLE_GOOGLE:
        for m in (data.get("models", []) if isinstance(data, dict) else []):
            name = m.get("name", "")
            methods = m.get("supportedGenerationMethods", []) or []
            if name and (not methods or "generateContent" in methods):
                ids.append(name.split("/")[-1])
    else:
        items = data.get("data", data) if isinstance(data, dict) else data
        if isinstance(items, dict):
            items = items.get("models", items.get("data", []))
        for m in (items or []):
            mid = m.get("id") or m.get("name") if isinstance(m, dict) else None
            if mid:
                ids.append(mid)
    return ids


def _gen_request(provider: Provider, key: str, model: str):
    """Готовит (url, headers, json) для мини-запроса генерации под стиль провайдера."""
    headers = auth_headers(provider, key)
    if provider.api_style == STYLE_ANTHROPIC:
        return provider.gen_url, headers, {
            "model": model, "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}
    if provider.api_style == STYLE_GOOGLE:
        url = f"{provider.gen_url}/models/{model}:generateContent"
        return url, headers, {
            "contents": [{"parts": [{"text": "hi"}]}],
            "generationConfig": {"maxOutputTokens": 1}}
    if provider.api_style == STYLE_COHERE:
        return provider.gen_url, headers, {
            "model": model, "max_tokens": 1,
            "messages": [{"role": "user", "content": "hi"}]}
    # openai-совместимый
    return provider.gen_url, headers, {
        "model": model, "max_tokens": 1,
        "messages": [{"role": "user", "content": "hi"}]}


async def check_funds(client: httpx.AsyncClient, provider: Provider, key: str) -> dict:
    """Наличие средств через реальный запрос генерации (max_tokens=1).

    Это ПЛАТНО (микросписание), поэтому вызывается только для активного ключа
    и только там, где нет точного API баланса.
    """
    if not provider.gen_url:
        return {"balance": None, "balance_currency": None, "balance_status": "unsupported",
                "balance_note": "У этого провайдера нет API баланса, а генерация не настроена."}

    models = await list_models(client, provider, key)
    if not models:
        return {"balance": None, "balance_currency": None, "balance_status": "unknown",
                "balance_note": "Не удалось получить список моделей для пробного запроса."}

    last_note = "Пробный запрос не дал однозначного ответа."
    for model in models[:MAX_MODEL_TRIES]:
        url, headers, payload = _gen_request(provider, key, model)
        try:
            r = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError:
            last_note = "Сбой сети при пробном запросе."
            continue

        text = response_error(r)
        if r.is_success:
            return {"balance": None, "balance_currency": None,
                    "balance_status": "funds_present",
                    "balance_note": f"Пробный запрос ({model}) прошёл — средства/квота есть "
                                    f"(точная сумма через API недоступна)."}
        code = r.status_code
        if code == 402 or _quota(text):
            return {"balance": None, "balance_currency": None,
                    "balance_status": "no_funds",
                    "balance_note": "Пробный запрос отклонён из-за средств/квоты — денег нет."}
        if code in (401, 403) and _auth_fail(text):
            return {"balance": None, "balance_currency": None, "balance_status": "error",
                    "balance_note": "Ключ отклонён на пробном запросе."}
        if "location is not supported" in text or "user location" in text:
            return {"balance": None, "balance_currency": None, "balance_status": "unknown",
                    "balance_note": "Пробный запрос заблокирован по региону — наличие средств неизвестно."}
        if code == 429:
            last_note = "Лимит запросов — наличие средств по этому коду не определить."
            continue
        # Модель не подошла (не чат / нет доступа) — пробуем следующую.
        last_note = f"Модель {model} не подошла (HTTP {code}); пробую другую."

    return {"balance": None, "balance_currency": None, "balance_status": "unknown",
            "balance_note": last_note}


async def check_balance(client: httpx.AsyncClient, provider: Provider, key: str,
                        probe_status: str = "valid") -> dict:
    """Баланс: точная сумма через API, иначе — наличие средств пробным запросом."""
    # Авторизационная проба уже сказала «квота/оплата» — деньги точно кончились.
    if probe_status == "no_credits":
        return {"balance": None, "balance_currency": None, "balance_status": "no_funds",
                "balance_note": "Провайдер сообщил об исчерпанной квоте/оплате."}

    if not provider.balance_url:
        return await check_funds(client, provider, key)

    try:
        response = await client.get(provider.balance_url, headers=auth_headers(provider, key))
    except httpx.TimeoutException:
        return {"balance": None, "balance_currency": None, "balance_status": "timeout",
                "balance_note": "Endpoint баланса не ответил вовремя."}
    except httpx.HTTPError:
        return {"balance": None, "balance_currency": None, "balance_status": "error",
                "balance_note": "Не удалось подключиться к endpoint баланса."}

    if not response.is_success:
        _, message = classify_auth(response)
        return {"balance": None, "balance_currency": None, "balance_status": "error",
                "balance_note": f"Баланс не получен: HTTP {response.status_code}. {message}"}

    try:
        data = response.json()
    except ValueError:
        return {"balance": None, "balance_currency": None, "balance_status": "error",
                "balance_note": "Ответ endpoint баланса не распознан."}

    if provider.balance_kind == BALANCE_OPENROUTER:
        payload = data.get("data", {}) if isinstance(data, dict) else {}
        limit = _num(payload.get("limit"))
        remaining = _num(payload.get("limit_remaining"))
        usage = _num(payload.get("usage"))
        if remaining is not None:
            return {"balance": round(remaining, 4), "balance_currency": "USD",
                    "balance_status": "available",
                    "balance_note": f"Остаток лимита ключа. Потрачено всего: ${usage:.4f}."
                    if usage is not None else "Остаток лимита ключа, USD."}
        if limit is None:
            note = "Лимит ключа не задан (безлимит)."
            if usage is not None:
                note += f" Потрачено всего: ${usage:.4f}."
            return {"balance": None, "balance_currency": "USD",
                    "balance_status": "unlimited", "balance_note": note}
        return {"balance": None, "balance_currency": "USD", "balance_status": "unknown",
                "balance_note": "Ответ не содержит остатка кредитов."}

    if provider.balance_kind == BALANCE_DEEPSEEK:
        infos = data.get("balance_infos", []) if isinstance(data, dict) else []
        available = data.get("is_available") if isinstance(data, dict) else None
        if isinstance(infos, list) and infos:
            # Берём первую валюту (обычно одна — CNY или USD).
            first = infos[0]
            total = _num(first.get("total_balance"))
            currency = first.get("currency")
            note = "Доступно для запросов." if available else "Баланс есть, но is_available=false."
            return {"balance": round(total, 4) if total is not None else None,
                    "balance_currency": currency, "balance_status": "available",
                    "balance_note": note}
        return {"balance": None, "balance_currency": None, "balance_status": "unknown",
                "balance_note": "Ответ не содержит данных о балансе."}

    return {"balance": None, "balance_currency": None, "balance_status": "unknown",
            "balance_note": "Формат баланса не распознан."}


# Провайдер определён (ключ точно принадлежит ему, даже если без прав/квоты).
MATCH_STATUSES = {"valid", "rate_limited", "no_credits", "forbidden", "region_blocked"}
# Ключ рабочий — имеет смысл спрашивать баланс.
ACTIVE_STATUSES = {"valid", "rate_limited", "no_credits"}
# Приоритет при выборе лучшего совпадения.
_RANK = {"valid": 0, "rate_limited": 1, "no_credits": 2,
         "region_blocked": 3, "forbidden": 4}


def _skip_balance(note: str) -> dict:
    return {"balance": None, "balance_currency": None, "balance_status": "skipped",
            "balance_note": note}


async def resolve_key(client: httpx.AsyncClient, key: str, provider_id: str) -> dict:
    """Определяет провайдера (при auto — перебором всех) и собирает полный результат."""
    if provider_id != "auto":
        provider = PROVIDER_BY_ID.get(provider_id)
        if provider is None:
            raise HTTPException(status_code=400, detail="Неизвестный провайдер.")
        probe = await probe_provider(client, provider, key)
        st = probe["status"]
        balance = (await check_balance(client, provider, key, st)
                   if st in ACTIVE_STATUSES or st == "no_credits"
                   else _skip_balance("Баланс не проверялся: ключ не активен у провайдера."))
        return {**probe, **balance, "provider_name": provider.name,
                "key_preview": mask_key(key)}

    # auto: прогоняем ключ по всем провайдерам параллельно.
    order = candidate_order(key)
    preferred_ids = {p.id for p in order if any(key.startswith(pre) for pre in p.prefixes)}
    probes = await asyncio.gather(*(probe_provider(client, p, key) for p in order))
    matched = [p for p in probes if p["status"] in MATCH_STATUSES]

    if not matched:
        # Ключ нигде не принят. Но если префикс однозначно указывает на провайдера,
        # а его проба сказала «недействителен» — атрибутируем провайдера.
        prefix_invalid = [p for p in probes
                          if p["provider"] in preferred_ids and p["status"] == "invalid"]
        if len(prefix_invalid) == 1:
            p = prefix_invalid[0]
            return {**p, **_skip_balance("Баланс не проверялся: ключ недействителен."),
                    "key_preview": mask_key(key)}
        return {"status": "unknown_provider", "provider": None,
                "provider_name": "Провайдер не определён", "key_preview": mask_key(key),
                "message": "Ключ не принят ни одним из проверенных провайдеров "
                           "(возможно, он отозван или это не LLM-ключ).",
                "status_code": None, **_skip_balance("Баланс не проверялся."),
                "candidates": []}

    matched.sort(key=lambda r: _RANK.get(r["status"], 9))
    best = matched[0]
    provider = PROVIDER_BY_ID[best["provider"]]

    balance = (await check_balance(client, provider, key, best["status"])
               if best["status"] in ACTIVE_STATUSES or best["status"] == "no_credits"
               else _skip_balance("Баланс не проверялся: ключ не активен."))

    result = {**best, **balance, "provider_name": provider.name,
              "key_preview": mask_key(key)}
    if len(matched) > 1:
        result["candidates"] = [{"provider": a["provider"],
                                 "provider_name": a.get("provider_name"),
                                 "status": a["status"]} for a in matched]
    return result


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/providers")
async def list_providers():
    return [{"id": p.id, "name": p.name} for p in PROVIDERS]


@app.post("/api/check")
async def check(body: CheckRequest):
    key = body.key.strip()
    if not key or any(char.isspace() for char in key):
        raise HTTPException(status_code=400, detail="Ключ пустой или содержит пробелы.")

    limits = httpx.Limits(max_connections=MAX_CONCURRENT_PROBES)
    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT, follow_redirects=False,
                                 limits=limits, headers=CLIENT_HEADERS) as client:
        return await resolve_key(client, key, body.provider)


@app.post("/api/scan")
async def scan(body: ScanRequest):
    """Массовая проверка: список ключей -> провайдер + активность + баланс для каждого."""
    keys = []
    seen = set()
    for raw in body.keys:
        k = (raw or "").strip()
        if not k or any(c.isspace() for c in k) or k in seen:
            continue
        seen.add(k)
        keys.append(k)

    if not keys:
        raise HTTPException(status_code=400, detail="Не найдено ни одного валидного ключа.")

    limits = httpx.Limits(max_connections=MAX_CONCURRENT_PROBES)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_PROBES)

    async with httpx.AsyncClient(timeout=PROBE_TIMEOUT, follow_redirects=False,
                                 limits=limits, headers=CLIENT_HEADERS) as client:
        async def one(k: str) -> dict:
            async with semaphore:
                return await resolve_key(client, k, "auto")

        results = await asyncio.gather(*(one(k) for k in keys))

    return {"count": len(results), "results": results}


if __name__ == "__main__":
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
