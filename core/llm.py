"""模型适配层（P2）：统一接口，后端与模型名全部来自 config/analysis.json。

原则：
- 领域层只依赖 LLMBackend.chat_json()，不感知协议细节；
- 不硬编码模型名，换后端/换模型只改配置；
- 仅用标准库，不新增依赖。
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "analysis.json"


@dataclass
class LLMResponse:
    text: str
    seconds: float
    backend: str
    model: str
    usage: dict | None = None


class LLMCallError(RuntimeError):
    """调用失败（网络/超时/服务端错误），code 用于失败分类统计。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class LLMConfigError(RuntimeError):
    """配置缺失（例如 API key 未设置），由调用方决定提示并回退。"""



def load_config(path: Path | str | None = None) -> dict:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


class LLMBackend:
    name = "base"
    model = ""

    def chat_json(self, system: str, user: str) -> LLMResponse:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, cfg: dict):
        self.base_url = str(cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        self.chat_path = str(cfg.get("chat_path", "/api/chat"))
        self.model = str(cfg.get("model", ""))
        self.temperature = cfg.get("temperature", 0)
        self.num_ctx = cfg.get("num_ctx", 4096)
        self.timeout = max(1, int(cfg.get("timeout_seconds", 180)))
        # 单条墙钟硬上限：即使连接/读取未按预期超时，到点也必须中断并继续下一条
        self.max_wall = max(1, int(cfg.get("max_wall_seconds", self.timeout)))

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + self.chat_path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        result: dict = {}

        def _worker() -> None:
            try:
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                result["resp"] = resp
                result["raw"] = resp.read()
            except BaseException as e:  # 在线程内统一收集，主线程负责分类
                result["err"] = e

        worker = threading.Thread(target=_worker, name="ollama-call", daemon=True)
        worker.start()
        worker.join(self.max_wall)
        if worker.is_alive():
            # 到点强制中断：关闭连接，避免服务器继续生成；主线程立即返回，批次继续
            resp = result.get("resp")
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            raise LLMCallError("wall_timeout", f"单条超过 {self.max_wall}s，已中断并继续")

        err = result.get("err")
        if isinstance(err, LLMCallError):
            raise err
        if isinstance(err, urllib.error.HTTPError):
            detail = ""
            try:
                detail = err.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            raise LLMCallError(f"http_{err.code}", f"HTTP {err.code}: {detail}") from err
        if isinstance(err, urllib.error.URLError):
            if isinstance(err.reason, TimeoutError):
                raise LLMCallError("timeout", f"超过 {self.timeout}s") from err
            raise LLMCallError("connection_error", str(err.reason)) from err
        if isinstance(err, TimeoutError):
            raise LLMCallError("timeout", f"超过 {self.timeout}s") from err
        if err is not None:
            raise LLMCallError("unexpected", f"{type(err).__name__}: {err}") from err

        raw = result["raw"].decode("utf-8", "ignore")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMCallError("bad_response", f"响应不是 JSON：{e}") from e

    def chat_json(self, system: str, user: str) -> LLMResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        t0 = time.perf_counter()
        body = self._post(payload)
        elapsed = time.perf_counter() - t0
        text = (body.get("message") or {}).get("content") or ""
        if not text:
            raise LLMCallError("empty_output", "模型返回空内容")
        return LLMResponse(text=text, seconds=elapsed, backend=self.name, model=self.model)


class OpenAICompatBackend(LLMBackend):
    """OpenAI 兼容后端（Groq / Gemini OpenAI 兼容层 / OpenRouter / GitHub Models 等）。

    配置优先级：环境变量 > config/analysis.json > 默认值。
    - base_url：FEEDBACKLENS_API_BASE 或 backends.openai_compat.base_url
    - model：   FEEDBACKLENS_API_MODEL 或 backends.openai_compat.model
    - api_key：api_key_env 指定的环境变量（默认 FEEDBACKLENS_API_KEY）或 api_key 字段
    密钥只留在内存，不写库、不打印；缺失时由 get_backend 提示并回退本地模型。
    """

    name = "openai_compat"

    RETRIABLE = {"http_429", "http_500", "http_502", "http_503", "http_504",
                 "connection_error", "timeout"}

    def __init__(self, cfg: dict):
        self.base_url = str(os.environ.get("FEEDBACKLENS_API_BASE")
                            or cfg.get("base_url") or "").strip().rstrip("/")
        chat_path = str(cfg.get("chat_path") or "/v1/chat/completions").strip()
        self.chat_path = chat_path if chat_path.startswith("/") else "/" + chat_path
        # base_url 已经带 /v1 时避免拼成 /v1/v1/...
        if self.base_url.endswith("/v1") and self.chat_path.startswith("/v1/"):
            self.chat_path = self.chat_path[3:]
        self.model = str(os.environ.get("FEEDBACKLENS_API_MODEL")
                         or cfg.get("model") or "").strip()
        self.api_key_env = str(cfg.get("api_key_env") or "FEEDBACKLENS_API_KEY")
        self.api_key = str(os.environ.get(self.api_key_env)
                           or cfg.get("api_key") or "").strip()
        self.timeout = max(1, int(cfg.get("timeout_seconds", 60)))
        # 单条墙钟硬上限：线程到点即中断，批次继续下一条
        self.max_wall = max(1, int(cfg.get("max_wall_seconds", self.timeout)))
        self.max_retries = max(0, int(cfg.get("max_retries", 1)))
        self.retry_backoff = max(0.0, float(cfg.get("retry_backoff_seconds", 2)))
        self.temperature = cfg.get("temperature", 0)
        self.response_format = cfg.get("response_format", "json_object")
        # Groq 前置 Cloudflare 会拦截 urllib 默认 UA（error 1010 = 403），
        # 这里给一个明确的客户端标识；cfg.headers 仍可覆盖。
        self.headers = {"User-Agent": "FeedbackLensAI/0.1 (local evaluation client)"}
        self.headers.update({str(k): str(v) for k, v in (cfg.get("headers") or {}).items()})
        # 调用统计（只计数，不含任何请求内容或密钥），供对比脚本汇报额度消耗
        self.stats: dict[str, int] = {}

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model and self.api_key)

    def config_hint(self) -> str:
        missing = []
        if not self.base_url:
            missing.append("base_url（FEEDBACKLENS_API_BASE）")
        if not self.model:
            missing.append("model（FEEDBACKLENS_API_MODEL）")
        if not self.api_key:
            missing.append(f"api_key（环境变量 {self.api_key_env}）")
        return "在线模型配置不完整，缺少：" + "、".join(missing)

    def _bump(self, key: str) -> None:
        self.stats[key] = int(self.stats.get(key, 0)) + 1

    def _post_once(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}", **self.headers}
        req = urllib.request.Request(
            self.base_url + self.chat_path,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        self._bump("http_requests")
        result: dict = {}

        def _worker() -> None:
            try:
                resp = urllib.request.urlopen(req, timeout=self.timeout)
                try:
                    result["raw"] = resp.read()
                finally:
                    resp.close()
            except BaseException as e:  # 线程内统一收集，主线程分类
                result["err"] = e

        worker = threading.Thread(target=_worker, name="openai-compat-call", daemon=True)
        worker.start()
        worker.join(self.max_wall)
        if worker.is_alive():
            raise LLMCallError("wall_timeout", f"单条超过 {self.max_wall}s，已中断并继续")

        err = result.get("err")
        if isinstance(err, urllib.error.HTTPError):
            self._bump(f"http_{err.code}")
            detail = ""
            try:
                detail = err.read().decode("utf-8", "ignore")[:200]
            except Exception:
                pass
            raise LLMCallError(f"http_{err.code}", f"HTTP {err.code}: {detail}") from err
        if isinstance(err, urllib.error.URLError):
            if isinstance(err.reason, TimeoutError):
                self._bump("timeout")
                raise LLMCallError("timeout", f"超过 {self.timeout}s") from err
            self._bump("connection_error")
            raise LLMCallError("connection_error", str(err.reason)) from err
        if isinstance(err, TimeoutError):
            raise LLMCallError("timeout", f"超过 {self.timeout}s") from err
        if err is not None:
            raise LLMCallError("unexpected", f"{type(err).__name__}: {err}") from err

        raw = result["raw"].decode("utf-8", "ignore")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise LLMCallError("bad_response", f"响应不是 JSON：{e}") from e

    def chat_json(self, system: str, user: str) -> LLMResponse:
        if not self.configured:
            raise LLMConfigError(self.config_hint())
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.response_format:
            payload["response_format"] = {"type": str(self.response_format)}

        t0 = time.perf_counter()
        body = None
        last_error: LLMCallError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                body = self._post_once(payload)
                break
            except LLMCallError as e:
                last_error = e
                elapsed = time.perf_counter() - t0
                if e.code not in self.RETRIABLE or attempt >= self.max_retries:
                    raise
                backoff = min(self.retry_backoff * (attempt + 1),
                              max(0.0, self.max_wall - elapsed))
                if backoff <= 0:
                    raise
                self._bump("retries")
                time.sleep(backoff)
        if body is None:
            raise last_error or LLMCallError("empty_output", "模型返回空内容")

        elapsed = time.perf_counter() - t0
        choice = (body.get("choices") or [{}])[0] if body.get("choices") else {}
        message = choice.get("message") or {}
        text = message.get("content") or choice.get("text") or ""
        if isinstance(text, list):  # 少数兼容层返回分段内容
            text = "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p)
                           for p in text)
        text = _strip_code_fences(str(text))
        if not text.strip():
            raise LLMCallError("empty_output", "模型返回空内容")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else None
        if usage:
            for src, dst in (("prompt_tokens", "prompt_tokens"),
                             ("completion_tokens", "completion_tokens"),
                             ("total_tokens", "total_tokens")):
                try:
                    self.stats[dst] = int(self.stats.get(dst, 0)) + int(usage.get(src) or 0)
                except (TypeError, ValueError):
                    pass
        return LLMResponse(text=text, seconds=elapsed, backend=self.name,
                           model=self.model, usage=usage)


def _strip_code_fences(text: str) -> str:
    """去掉 ```json ... ``` 包裹；部分免费模型不支持 json_object 时会加围栏。"""
    s = text.strip()
    if not s.startswith("```"):
        return s
    lines = s.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def get_backend(config: dict, strict: bool = False) -> LLMBackend:
    """按 active_backend 构造后端。

    strict=True 时配置缺失直接抛 LLMConfigError（对比脚本用，避免静默跑成本地模型）；
    strict=False 时打印明确提示并回退本地 ollama（应用默认行为）。
    """
    name = config.get("active_backend", "ollama")
    backends = config.get("backends", {})
    if name == "ollama":
        return OllamaBackend(backends.get("ollama", {}))
    if name == "openai_compat":
        bcfg = backends.get("openai_compat", {})
        if not bcfg.get("enabled"):
            raise ValueError("openai_compat 后端未启用：请把 config/analysis.json 的 "
                             "backends.openai_compat.enabled 设为 true")
        backend = OpenAICompatBackend(bcfg)
        if not backend.configured:
            hint = backend.config_hint()
            if strict:
                raise LLMConfigError(hint)
            import sys
            print(f"[warn] {hint}；已回退本地 ollama 后端", file=sys.stderr)
            return OllamaBackend(backends.get("ollama", {}))
        return backend
    raise ValueError(f"未知后端：{name}")


def list_ollama_models(base_url: str = "http://localhost:11434", timeout: int = 5) -> list[str]:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", "ignore"))
        return [m.get("name", "") for m in body.get("models", [])]
    except Exception:
        return []
