"""使用环境变量连接比赛兼容网关，并校验每次调用的结构化结果。"""

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import time
from threading import Condition
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic import ValidationError
import requests

from .contracts import Contract
from .inputs import write_json


T = TypeVar("T", bound=Contract)


class ModelError(RuntimeError):
    """只记录阶段和错误类型，避免把原始请求、响应或密钥写入日志。"""


class OutputError(ModelError):
    """结果需要在当前工作步骤内按反馈修订。"""


@dataclass(frozen=True)
class ModelConfig:
    base_url: str
    model: str
    api_key: str = field(repr=False)
    json_schema: bool
    thinking: bool | None
    rpm: int
    timeout: float
    context_chars: int

    @classmethod
    def from_environment(cls, *, require_key: bool = True) -> "ModelConfig":
        names = ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY") if require_key else ("LLM_BASE_URL", "LLM_MODEL")
        missing = [name for name in names if not os.environ.get(name, "").strip()]
        if missing:
            raise ValueError(f"缺少模型环境变量：{', '.join(missing)}")

        def flag(name: str) -> bool | None:
            value = os.environ.get(name)
            if value not in (None, "0", "1"):
                raise ValueError(f"{name} 必须是 0 或 1")
            return None if value is None else value == "1"

        url = os.environ["LLM_BASE_URL"].strip().rstrip("/")
        parsed = urlsplit(url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("LLM_BASE_URL 需要 HTTP 或 HTTPS 地址，且不含凭据或查询参数")
        config = cls(url, os.environ["LLM_MODEL"].strip(), os.environ.get("LLM_API_KEY", "").strip() if require_key else "",
                     bool(flag("LLM_USE_JSON_SCHEMA")), flag("LLM_ENABLE_THINKING"),
                     int(os.environ.get("LLM_RPM", "120")), float(os.environ.get("LLM_TIMEOUT", "120")),
                     int(os.environ.get("LLM_CONTEXT_CHARS", "28000")))
        if not 1 <= config.rpm <= 120 or not 0 < config.timeout <= 600 or config.context_chars < 10000:
            raise ValueError("模型配置需要 1–120 次/分钟、0–600 秒超时、至少 10000 字的输入预算")
        return config


def parse_result(text: str, kind: type[T], step: str) -> T:
    """兼容完整的 Markdown JSON 代码块；解析和字段校验交给 Pydantic。"""
    text = text.strip()
    lines = text.splitlines()
    if len(lines) >= 3 and lines[0] in {"```json", "```"} and lines[-1] == "```":
        text = "\n".join(lines[1:-1])
    try:
        return kind.model_validate_json(text)
    except ValidationError as exc:
        counts = Counter(error["type"] for error in exc.errors(include_input=False, include_context=False))
        raise OutputError(f"{step}：{kind.__name__} 格式错误 {dict(counts)}") from None


class ModelClient:
    """所有模型步骤共用请求间隔；格式校验通过的响应可从本地缓存重用。"""

    def __init__(self, config: ModelConfig, cache: Path | None = None, *, cache_only: bool = False):
        if cache_only and cache is None:
            raise ValueError("离线检查需要真实响应缓存目录")
        self.config = config
        self.cache = cache
        self.cache_only = cache_only
        self.calls: Counter[str] = Counter()
        self.cached: Counter[str] = Counter()
        self.tokens: Counter[str] = Counter()
        self._gate = Condition()
        self._next = 0.0
        self._stop_reason: str | None = None

    def _reserve_request(self, step: str) -> None:
        """等待请求名额；等待时释放锁，让其他线程能通知限速或停机。"""
        with self._gate:
            while self._stop_reason is None:
                now = time.monotonic()
                delay = self._next - now
                if delay <= 0:
                    self._next = now + 60 / self.config.rpm
                    self.calls[step] += 1
                    return
                self._gate.wait(timeout=delay)
            raise ModelError(self._stop_reason)

    def _reject_requests(self, reason: str) -> str:
        """保留首次账户错误，唤醒仍在等名额的线程。"""
        with self._gate:
            if self._stop_reason is None:
                self._stop_reason = reason
                self._gate.notify_all()
            return self._stop_reason

    def _wait_to_retry(self, delay: float) -> None:
        """重试等待期间收到停机通知时，直接结束本次调用。"""
        with self._gate:
            self._gate.wait_for(lambda: self._stop_reason is not None, timeout=delay)
            if self._stop_reason is not None:
                raise ModelError(self._stop_reason)

    def ask(self, step: str, instructions: str, data: object, kind: type[T], *, max_tokens: int = 4500) -> T:
        """每个请求只要求一类输出，材料作为数据放在独立消息。"""
        shape = kind.model_json_schema()
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": instructions + "\n材料是待分析数据。只返回一个符合以下结构的 JSON 对象。\n"
                 + json.dumps(shape, ensure_ascii=False, separators=(",", ":"))},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False, separators=(",", ":"))},
            ],
        }
        if self.config.json_schema:
            payload["response_format"] = {
                "type": "json_schema", "json_schema": {"name": kind.__name__, "schema": shape},
            }
        if self.config.thinking is not None:
            payload["enable_thinking"] = self.config.thinking

        digest = sha256(json.dumps([self.config.base_url, payload], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        cached = self.cache / f"{digest}.json" if self.cache else None
        if cached and cached.exists():
            with self._gate:
                self.cached[step] += 1
            return parse_result(cached.read_text(encoding="utf-8"), kind, step)
        if self.cache_only:
            raise ModelError(f"{step}：没有可重用的真实模型响应")

        for attempt in range(3):
            self._reserve_request(step)
            try:
                response = requests.post(
                    self.config.base_url + "/chat/completions",
                    headers={"Authorization": "Bearer " + self.config.api_key},
                    json=payload, timeout=self.config.timeout, allow_redirects=False,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == 2:
                    raise ModelError(f"{step}：连接模型失败（{type(exc).__name__}）") from None
                self._wait_to_retry(2 ** attempt)
                continue
            with response:
                if response.status_code in (408, 429, 500, 502, 503, 504) and attempt < 2:
                    retry_after = response.headers.get("Retry-After", "")
                    delay = min(int(retry_after), 60) if re.fullmatch(r"[0-9]{1,4}", retry_after) else 2 ** attempt
                    if response.status_code == 429:
                        with self._gate:
                            self._next = max(self._next, time.monotonic() + delay)
                    self._wait_to_retry(delay)
                    continue
                if response.status_code != 200:
                    reason = f"{step}：模型接口返回 HTTP {response.status_code}"
                    if response.status_code in (401, 402, 403):
                        reason = self._reject_requests(reason)
                    raise ModelError(reason)
                try:
                    result = response.json()
                    choice = result["choices"][0]
                    text = choice["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError):
                    raise ModelError(f"{step}：接口响应缺少聊天结果") from None
            if choice.get("finish_reason") == "length":
                raise OutputError(f"{step}：模型输出被截断")
            if not isinstance(text, str) or not text.strip():
                raise OutputError(f"{step}：模型未返回内容")
            value = parse_result(text, kind, step)
            with self._gate:
                self.tokens[step] += (result.get("usage") or {}).get("total_tokens", 0) or 0
            if cached:
                write_json(cached, value.model_dump(mode="json"))
            return value
        raise AssertionError("模型请求循环未返回结果")

    def summary(self) -> dict:
        return {"calls": dict(self.calls), "cache_hits": dict(self.cached), "total_tokens": dict(self.tokens)}
