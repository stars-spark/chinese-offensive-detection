"""D8 DeepSeek API 零样本试点。

只跑 COLD 验证集去重后按种子 0 抽出的 1000 条中的前 200 条。提示词从
``llm_zeroshot`` 原样导入，不复制改写。请求关闭思考模式，温度 0，最多
输出 1 个词元，取第一个词元的 top_logprobs 计算“是”的概率。

用法：
    export DEEPSEEK_API_KEY=...
    python src/llm_api.py --pilot
    python src/llm_api.py --pilot --dry-run   # 不调用 API，只核对抽样

密钥只从环境变量读，不写文件、不进日志；请求头从不记录。
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from data import deduplicate_cold_splits, load_splits
from engine import classification_metrics
from llm_zeroshot import PROMPTS, SYSTEM_PROMPT, build_messages, clean_comment
from result_io import resolve_results_dir, save_json, setup_logger

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_NAME = "试点"
DEFAULT_LIMIT = 200
POOL_N = 1000
MAX_TOP_LOGPROBS = 20
MISSING_LOGPROB = -30.0
MAX_WORKERS = 4
MAX_RETRIES = 3
STATUSES = ("ok", "partial", "no_prob", "blocked", "error")
PARQUET_COLUMNS = [
    "index", "label", "prob_pos", "status", "out_token",
    "prompt_tokens", "completion_tokens", "latency_ms",
    "error_code", "error_type", "error_message",
]
BEIJING = timezone(timedelta(hours=8))

PRICING_SOURCE_CN = "https://api-docs.deepseek.com/zh-cn/quick_start/pricing"
PRICING_SOURCE_EN = "https://api-docs.deepseek.com/quick_start/pricing"
# 页面单位：每百万 tokens。页面原文见 PRICING_RAW_*。
PRICING_CN = {
    "input_cache_hit_off_peak": 0.02,
    "input_cache_hit_peak": 0.04,
    "input_cache_miss_off_peak": 1.0,
    "input_cache_miss_peak": 2.0,
    "output_off_peak": 4.0,
    "output_peak": 8.0,
}
PRICING_USD = {
    "input_cache_hit_off_peak": 0.003,
    "input_cache_hit_peak": 0.006,
    "input_cache_miss_off_peak": 0.15,
    "input_cache_miss_peak": 0.3,
    "output_off_peak": 0.6,
    "output_peak": 1.2,
}
PRICING_RAW_CN = [
    "百万tokens输入（缓存命中）：空闲时段 0.02元，高峰时段 0.04元",
    "百万tokens输入（缓存未命中）：空闲时段 1元，高峰时段 2元",
    "百万tokens输出：空闲时段 4元，高峰时段 8元",
]
PRICING_RAW_EN = [
    "1M INPUT TOKENS (CACHE HIT) | OFF-PEAK $0.003 | PEAK $0.006",
    "1M INPUT TOKENS (CACHE MISS) | OFF-PEAK $0.15 | PEAK $0.3",
    "1M OUTPUT TOKENS | OFF-PEAK $0.6 | PEAK $1.2",
]
OFF_PEAK_NOTE = (
    "页面注明：空闲时段价格为高峰时段的一半；北京时间周一至周五（不含中国"
    "法定节假日）09:00–12:00、14:00–18:00 为高峰，即 UTC 01:00–04:00、"
    "06:00–10:00；其余时段包括周末全天为谷时。"
)


def is_peak(when: datetime) -> bool:
    """按 UTC 判断是否处于 DeepSeek 高峰时段（不含中国法定节假日）。"""
    when = when.astimezone(timezone.utc)
    if when.weekday() >= 5:
        return False
    return 1 <= when.hour < 4 or 6 <= when.hour < 10


def next_offpeak(when: datetime) -> datetime:
    """返回不早于 ``when`` 的下一个谷时起点。"""
    when = when.astimezone(timezone.utc)
    while is_peak(when):
        if 1 <= when.hour < 4:
            when = when.replace(hour=4, minute=0, second=0, microsecond=0)
        elif 6 <= when.hour < 10:
            when = when.replace(hour=10, minute=0, second=0, microsecond=0)
        else:
            when += timedelta(minutes=1)
    return when


def redact_secret(value: Any, secret: str | None) -> str:
    """把字符串里的密钥替换成 ``***``；密钥为空时原样返回。"""
    text = str(value)
    if secret:
        text = text.replace(secret, "***")
    return text


class RedactFilter(logging.Filter):
    """日志过滤器：任何 handler 写盘前再兜底替换一次密钥。"""

    def __init__(self, secret: str | None):
        super().__init__()
        self.secret = secret or ""

    def filter(self, record: logging.LogRecord) -> bool:
        if self.secret:
            record.msg = redact_secret(record.getMessage(), self.secret)
            record.args = ()
        return True


def make_logger(path: Path, secret: str | None = None) -> logging.Logger:
    """建日志，并给每个 handler 加脱敏过滤器。"""
    logger = setup_logger(path)
    flt = RedactFilter(secret)
    for handler in logger.handlers:
        handler.addFilter(flt)
    return logger


def load_api_key(environ: dict[str, str] | None = None) -> str:
    """只从环境变量读密钥；读不到就停下，不打印、不落盘。"""
    env = os.environ if environ is None else environ
    key = str(env.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise SystemExit("未取到密钥：环境变量 DEEPSEEK_API_KEY 未设置或为空")
    return key


@dataclass
class ScoreResult:
    status: str
    prob_pos: float | None = None
    out_token: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: int = 0
    error_code: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    attempts: int = 1


class PeakPause(RuntimeError):
    """遇到高峰且选择暂停时抛出，由主循环收尾。"""


def prob_pos_from_top_logprobs(
    top_logprobs: Iterable[dict] | None,
) -> tuple[float | None, str]:
    """从第一个词元的候选里取“是”“否”的对数概率。

    返回 ``(prob_pos, status)``：两个都在为 ``ok``；只有一个时另一个按
    ``MISSING_LOGPROB`` 计，状态 ``partial``；两个都不在为 ``None``，
    状态 ``no_prob``。用 sigmoid 做两类 softmax，数值上等价且不易溢出。
    """
    yes = no = None
    for item in top_logprobs or []:
        try:
            token = str(item.get("token", "")).strip()
            logprob = float(item.get("logprob"))
        except (AttributeError, TypeError, ValueError):
            continue
        if not math.isfinite(logprob):
            continue
        if token == "是":
            yes = logprob if yes is None else max(yes, logprob)
        elif token == "否":
            no = logprob if no is None else max(no, logprob)
    if yes is None and no is None:
        return None, "no_prob"
    if yes is None:
        yes, status = MISSING_LOGPROB, "partial"
    elif no is None:
        no, status = MISSING_LOGPROB, "partial"
    else:
        status = "ok"
    diff = yes - no
    if diff >= 0:
        prob = 1.0 / (1.0 + math.exp(-diff))
    else:
        exp_diff = math.exp(diff)
        prob = exp_diff / (1.0 + exp_diff)
    return float(prob), status


RISK_MARKERS = (
    "content risk", "content_risk", "content filter", "content_filter",
    "risk control", "safety_check", "sensitive", "审核", "违规", "敏感",
)


def _is_risk_text(text: Any) -> bool:
    low = str(text).casefold()
    return any(marker in low for marker in RISK_MARKERS)


def _decode_error(raw: bytes | str) -> tuple[str, str | None, str | None]:
    """解出错误体里的 code/type；不保留可能带评论原文的 message。"""
    text = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
    try:
        body = json.loads(text)
    except Exception:
        body = None
    error = body.get("error") if isinstance(body, dict) else None
    code = error_type = None
    if isinstance(error, dict):
        code = str(error.get("code") or "")[:60] or None
        error_type = str(error.get("type") or "")[:60] or None
    return text, code, error_type


def classify_error(status: int, raw: bytes | str) -> tuple[str, str | None, str]:
    """返回 ``(error_code, error_type, 本地的简短类别)``，不返回原始 message。"""
    text, code, error_type = _decode_error(raw)
    risk = _is_risk_text(text) or _is_risk_text(code) or _is_risk_text(error_type)
    if risk:
        short = "content_risk"
    elif status == 429:
        short = "rate_limit"
    elif 500 <= status < 600:
        short = "server_error"
    elif status == 400:
        short = "http_400"
    elif status == 401:
        short = "unauthorized"
    elif status == 403:
        short = "forbidden"
    else:
        short = f"http_{status}"
    error_code = f"HTTP {status}" + (f"/{code}" if code else "")
    return error_code[:80], error_type, short


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_token(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)[:32]


def parse_completion(payload: Any, latency_ms: int = 0,
                     attempts: int = 1) -> ScoreResult:
    """把一次 200 响应解析成逐条结果；拦截也算 blocked。"""
    if not isinstance(payload, dict):
        return ScoreResult("error", latency_ms=latency_ms, error_code="BAD_JSON",
                           error_type="parse_error", error_message="bad_json",
                           attempts=attempts)
    if payload.get("error"):
        error = payload["error"]
        text = json.dumps(error, ensure_ascii=False)
        risk = _is_risk_text(text)
        error_type = (str(error.get("type") or "")[:60] or None
                      if isinstance(error, dict) else None)
        return ScoreResult("blocked" if risk else "error", latency_ms=latency_ms,
                           error_code="API_ERROR", error_type=error_type,
                           error_message="content_risk" if risk else "api_error",
                           attempts=attempts)
    choices = payload.get("choices") or []
    if not choices:
        return ScoreResult("error", latency_ms=latency_ms, error_code="EMPTY_CHOICES",
                           error_type="empty_choices", error_message="empty_choices",
                           attempts=attempts)
    choice = choices[0] or {}
    message = choice.get("message") or {}
    finish = str(choice.get("finish_reason") or "")
    if finish == "content_filter" or message.get("refusal"):
        return ScoreResult("blocked", latency_ms=latency_ms,
                           error_code="content_filter", error_type="content_filter",
                           error_message="content_risk", attempts=attempts)
    logprobs = (choice.get("logprobs") or {}).get("content") or []
    out_token = None
    top_logprobs = []
    if logprobs:
        first = logprobs[0] or {}
        out_token = first.get("token")
        top_logprobs = first.get("top_logprobs") or []
    elif message.get("content") is not None:
        out_token = message.get("content")
    prob, status = prob_pos_from_top_logprobs(top_logprobs)
    usage = payload.get("usage") or {}
    return ScoreResult(
        status=status,
        prob_pos=prob,
        out_token=_clean_token(out_token),
        prompt_tokens=_to_int(usage.get("prompt_tokens")),
        completion_tokens=_to_int(usage.get("completion_tokens")),
        latency_ms=int(latency_ms),
        attempts=attempts,
    )


class DeepSeekClient:
    """只用标准库 urllib 的 DeepSeek OpenAI 兼容客户端。"""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL,
                 endpoint: str = f"{DEFAULT_BASE_URL}/chat/completions",
                 timeout: float = 60.0, max_retries: int = MAX_RETRIES,
                 top_logprobs: int = MAX_TOP_LOGPROBS,
                 sleep_fn: Callable[[float], None] = time.sleep,
                 opener: Callable[..., Any] | None = None,
                 logger: logging.Logger | None = None):
        if not api_key:
            raise ValueError("api_key 不能为空")
        if not 0 <= max_retries <= MAX_RETRIES:
            raise ValueError(f"max_retries 必须在 0～{MAX_RETRIES}")
        if not 1 <= top_logprobs <= MAX_TOP_LOGPROBS:
            raise ValueError(f"top_logprobs 必须在 1～{MAX_TOP_LOGPROBS}")
        self.api_key = api_key
        self.model = model
        self.endpoint = endpoint
        self.timeout = timeout
        self.max_retries = max_retries
        self.top_logprobs = top_logprobs
        self.sleep_fn = sleep_fn
        self._opener = opener or urllib.request.urlopen
        self.logger = logger

    def _payload(self, messages: list[dict]) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 1,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "thinking": {"type": "disabled"},
            "stream": False,
        }

    def _post(self, payload: dict) -> tuple[int, bytes, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint, data=data, method="POST",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                # 请求头只在这里出现，从不写日志
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            response = self._opener(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            return int(exc.code), exc.read(), exc.headers
        try:
            status = int(getattr(response, "status", 200) or 200)
            return status, response.read(), getattr(response, "headers", None)
        finally:
            close = getattr(response, "close", None)
            if close is not None:
                close()

    @staticmethod
    def _retry_after(headers: Any) -> float | None:
        if not headers:
            return None
        try:
            value = headers.get("Retry-After")
            return max(0.0, float(value)) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _delay(self, attempt: int, headers: Any) -> float:
        delay = float(2 ** attempt)
        retry_after = self._retry_after(headers)
        if retry_after is not None:
            return max(delay, retry_after)
        return delay

    def _log_retry(self, reason: Any, attempt: int, delay: float) -> None:
        if self.logger is not None:
            self.logger.warning("%s，%.1f 秒后重试（%d/%d）", reason, delay,
                                attempt + 1, self.max_retries)

    def score(self, messages: list[dict]) -> ScoreResult:
        """发一条请求；429/5xx/网络错误按 1、2、4 秒退避，最多重试 3 次。"""
        payload = self._payload(messages)
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                status, raw, headers = self._post(payload)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                elapsed = int((time.perf_counter() - started) * 1000)
                if attempt < self.max_retries:
                    delay = self._delay(attempt, None)
                    self._log_retry(type(exc).__name__, attempt, delay)
                    self.sleep_fn(delay)
                    continue
                return ScoreResult("error", latency_ms=elapsed, error_code="NETWORK",
                                   error_type=type(exc).__name__,
                                   error_message="network_error",
                                   attempts=attempt + 1)
            elapsed = int((time.perf_counter() - started) * 1000)
            if status == 200:
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    return ScoreResult("error", latency_ms=elapsed, error_code="BAD_JSON",
                                       error_type="parse_error", error_message="bad_json",
                                       attempts=attempt + 1)
                return parse_completion(body, elapsed, attempt + 1)
            error_code, error_type, short = classify_error(status, raw)
            if status == 400 or short == "content_risk":
                return ScoreResult("blocked", latency_ms=elapsed, error_code=error_code,
                                   error_type=error_type, error_message=short,
                                   attempts=attempt + 1)
            if status == 429 or 500 <= status < 600:
                if attempt < self.max_retries:
                    delay = self._delay(attempt, headers)
                    self._log_retry(f"HTTP {status}", attempt, delay)
                    self.sleep_fn(delay)
                    continue
                return ScoreResult("error", latency_ms=elapsed, error_code=error_code,
                                   error_type=error_type, error_message=short,
                                   attempts=attempt + 1)
            return ScoreResult("error", latency_ms=elapsed, error_code=error_code,
                               error_type=error_type, error_message=short,
                               attempts=attempt + 1)
        return ScoreResult("error", latency_ms=int((time.perf_counter() - started) * 1000),
                           error_code="UNKNOWN", error_type="unknown",
                           error_message="unknown", attempts=self.max_retries + 1)


def empty_results() -> pd.DataFrame:
    """空结果表，列与正式 parquet 完全一致；不含评论原文。"""
    dtypes = {
        "index": "int64", "label": "int64", "prob_pos": "float64",
        "status": "object", "out_token": "object",
        "prompt_tokens": "Int64", "completion_tokens": "Int64",
        "latency_ms": "int64", "error_code": "object",
        "error_type": "object", "error_message": "object",
    }
    return pd.DataFrame({column: pd.Series(dtype=dtype)
                         for column, dtype in dtypes.items()})


def load_results(path: Path) -> pd.DataFrame:
    """读已有逐条结果；不存在时返回空表，供断点续跑。"""
    if not path.exists():
        return empty_results()
    frame = pd.read_parquet(path)
    for column in PARQUET_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    return frame[PARQUET_COLUMNS].copy()


def write_results(path: Path, rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """原子写 parquet；只保留固定列，绝不写评论原文。"""
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    for column in PARQUET_COLUMNS:
        if column not in frame:
            frame[column] = pd.NA
    frame = frame[PARQUET_COLUMNS].copy()
    frame["index"] = pd.to_numeric(frame["index"], errors="raise").astype("int64")
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int64")
    frame["prob_pos"] = pd.to_numeric(frame["prob_pos"], errors="coerce").astype("float64")
    for column in ("prompt_tokens", "completion_tokens"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    frame["latency_ms"] = (pd.to_numeric(frame["latency_ms"], errors="coerce")
                           .fillna(0).astype("int64"))
    for column in ("status", "out_token", "error_code", "error_type", "error_message"):
        frame[column] = frame[column].astype("string")
    frame = (frame.sort_values("index").drop_duplicates("index", keep="last")
             .reset_index(drop=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)
    return frame


def pilot_sample(limit: int = DEFAULT_LIMIT, seed: int = 0,
                 pool_n: int = POOL_N) -> tuple[pd.DataFrame, dict]:
    """按 ``llm_zeroshot --select`` 的同一行代码抽 1000 条，再取前 200 条。

    ``index`` 是这 1000 条池子里的位置（0 起），不是 COLD 原始行号；这样
    以后把试点扩到 1000 条时，已有结果可以按 index 直接跳过。
    """
    if limit <= 0 or pool_n <= 0:
        raise ValueError("limit 与 pool_n 必须为正整数")
    splits = load_splits("cold")
    dedup, removed = deduplicate_cold_splits(splits)
    val = dedup["val"]
    actual_pool = min(int(pool_n), len(val))
    pool = val.sample(n=actual_pool, random_state=int(seed)).reset_index(drop=True)
    n = min(int(limit), actual_pool)
    sample = pool.head(n).copy()
    sample.insert(0, "index", np.arange(n, dtype=np.int64))
    meta = {
        "source": "COLD 官方 dev（跨划分去重后）",
        "method": (f"val.sample(n={actual_pool}, random_state={int(seed)})"
                   f".reset_index(drop=True).head({n})"),
        "seed": int(seed),
        "val_n": int(len(val)),
        "pool_n": int(actual_pool),
        "pilot_n": int(n),
        "dedup_removed": {key: int(value) for key, value in removed.items()},
        "label_counts": {
            str(key): int(value)
            for key, value in sample["label"].value_counts().sort_index().items()
        },
    }
    return sample[["index", "text", "label"]], meta


def result_row(index: int, label: int, result: ScoreResult) -> dict:
    """把一次调用转成 parquet 的一行；字段固定，不含评论。"""
    return {
        "index": int(index),
        "label": int(label),
        "prob_pos": result.prob_pos,
        "status": result.status,
        "out_token": result.out_token,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "latency_ms": int(result.latency_ms),
        "error_code": result.error_code,
        "error_type": result.error_type,
        "error_message": result.error_message,
    }


def make_peak_check(action: str, logger: logging.Logger | None = None):
    """返回请求前的高峰检查；``pause`` 抛 PeakPause，``wait`` 睡到谷时。"""
    def check() -> None:
        now = datetime.now(timezone.utc)
        if not is_peak(now):
            return
        if action == "wait":
            target = next_offpeak(now)
            seconds = max(1.0, (target - now).total_seconds())
            if logger is not None:
                logger.warning("高峰时段，暂停 %.0f 秒到 %s", seconds,
                               target.isoformat(timespec="minutes"))
            time.sleep(seconds)
            return
        raise PeakPause(
            f"高峰时段（UTC {now:%H:%M}），已暂停；稍后用同一命令续跑")
    return check


def score_one(client: DeepSeekClient, index: int, text: str, label: int,
              logger: logging.Logger | None = None,
              peak_check: Callable[[], None] | None = None) -> dict:
    """一条评论：先建消息，再请求；任何异常都落成 error，不抛评论内容。"""
    if peak_check is not None:
        peak_check()
    try:
        messages = build_messages("P2", clean_comment(text))
        result = client.score(messages)
    except PeakPause:
        raise
    except Exception as exc:
        result = ScoreResult("error", latency_ms=0, error_code="CLIENT_ERROR",
                             error_type=type(exc).__name__,
                             error_message="client_error")
    if logger is not None:
        probability = "NA" if result.prob_pos is None else f"{result.prob_pos:.6f}"
        logger.info("index=%d label=%d status=%s prob=%s out=%r tokens=%s/%s %dms",
                    index, label, result.status, probability, result.out_token,
                    result.prompt_tokens, result.completion_tokens,
                    result.latency_ms)
        if result.status in ("blocked", "error"):
            logger.warning("index=%d %s code=%s type=%s msg=%s", index,
                           result.status, result.error_code, result.error_type,
                           result.error_message)
    return result_row(index, label, result)


def run_pending(client: DeepSeekClient, sample: pd.DataFrame, out_parquet: Path,
                workers: int = MAX_WORKERS, flush_every: int = 10,
                logger: logging.Logger | None = None,
                peak_check: Callable[[], None] | None = None) -> pd.DataFrame:
    """跑还没有结果的 index；并发不超过 4，每 10 条写一次 parquet。"""
    existing = load_results(out_parquet)
    done = {int(value) for value in existing["index"]} if len(existing) else set()
    pending = [
        (int(row.index), str(row.text), int(row.label))
        for row in sample.itertuples(index=False)
        if int(row.index) not in done
    ]
    if not pending:
        return existing

    new_rows: list[dict] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {
        pool.submit(score_one, client, index, text, label, logger, peak_check):
            (index, label)
        for index, text, label in pending
    }
    interrupted: Exception | None = None
    try:
        for future in as_completed(futures):
            index, label = futures[future]
            try:
                row = future.result()
            except PeakPause as exc:
                interrupted = exc
                for pending_future in futures:
                    pending_future.cancel()
                break
            except Exception as exc:
                row = result_row(index, label, ScoreResult(
                    "error", error_code="CLIENT_ERROR",
                    error_type=type(exc).__name__, error_message="client_error"))
            new_rows.append(row)
            if flush_every and len(new_rows) % flush_every == 0:
                write_results(out_parquet, pd.concat(
                    [existing, pd.DataFrame(new_rows)], ignore_index=True))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    if interrupted is not None:
        # 把已经完成但还没被 as_completed 取走的行也收进来
        for future in futures:
            if future.cancelled():
                continue
            try:
                new_rows.append(future.result(timeout=0))
            except Exception:
                continue
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    combined = write_results(out_parquet, combined)
    if interrupted is not None:
        raise interrupted
    return combined


def cost_per_1k(prompt_tokens: float, completion_tokens: float,
                input_per_million: float, output_per_million: float) -> float:
    """按每百万 tokens 单价算 1000 条的费用。"""
    return float(
        (float(prompt_tokens) * float(input_per_million)
         + float(completion_tokens) * float(output_per_million)) / 1_000_000 * 1000
    )


def release_verdict(status_counts: dict[str, int], n: int) -> dict:
    """两条放行条件，逐条给出是否满足；阈值不改。"""
    if n <= 0:
        raise ValueError("n 必须为正整数")
    blocked_error = int(status_counts.get("blocked", 0)) + int(status_counts.get("error", 0))
    has_prob = int(status_counts.get("ok", 0)) + int(status_counts.get("partial", 0))
    blocked_ratio = blocked_error / n
    has_prob_ratio = has_prob / n
    first = blocked_ratio <= 0.05 + 1e-12
    second = has_prob_ratio >= 0.95 - 1e-12
    return {
        "items": [
            {
                "name": "拦截与报错合计 ≤ 5%",
                "count": blocked_error,
                "ratio": blocked_ratio,
                "threshold": 0.05,
                "pass": bool(first),
            },
            {
                "name": "有概率的比例 ≥ 95%",
                "count": has_prob,
                "ratio": has_prob_ratio,
                "threshold": 0.95,
                "pass": bool(second),
            },
        ],
        "overall_pass": bool(first and second),
    }


def _mean_or_none(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else None


def summarize(frame: pd.DataFrame, meta: dict, args, started: datetime,
              finished: datetime) -> dict:
    """汇总试点 JSON；只照报，不做“有差异”判定。"""
    n = len(frame)
    if n <= 0:
        raise ValueError("没有可汇总的逐条结果")
    status_counts = {status: int((frame["status"] == status).sum())
                     for status in STATUSES}
    status_ratios = {status: status_counts[status] / n for status in STATUSES}
    has_prob = status_counts["ok"] + status_counts["partial"]
    blocked_error = status_counts["blocked"] + status_counts["error"]
    prompt_mean = _mean_or_none(frame["prompt_tokens"])
    completion_mean = _mean_or_none(frame["completion_tokens"])
    latency_mean = _mean_or_none(frame["latency_ms"])
    latency_ok = _mean_or_none(
        frame.loc[frame["status"].isin(["ok", "partial"]), "latency_ms"])
    latency_values = pd.to_numeric(frame["latency_ms"], errors="coerce").dropna()
    latency_p95 = (float(np.percentile(latency_values, 95))
                   if len(latency_values) else None)

    costs: dict[str, float | None] = {}
    if prompt_mean is not None and completion_mean is not None:
        for currency, prices in (("cny", PRICING_CN), ("usd", PRICING_USD)):
            for period in ("off_peak", "peak"):
                for cache in ("cache_miss", "cache_hit"):
                    costs[f"cost_per_1k_{currency}_{period}_{cache}"] = round(
                        cost_per_1k(prompt_mean, completion_mean,
                                    prices[f"input_{cache}_{period}"],
                                    prices[f"output_{period}"]), 6)

    labels = frame["label"].to_numpy(dtype=np.int64)
    probabilities = pd.to_numeric(frame["prob_pos"], errors="coerce").to_numpy(dtype=float)
    missing = np.isnan(probabilities)
    # 全量口径：被拦截或无概率按“判为攻击性”计。试点照此只报。
    predictions = np.where(missing, 1, (probabilities >= 0.5).astype(np.int64))
    metrics = classification_metrics(labels, predictions, ["安全", "攻击性"])
    metrics_payload = {
        "threshold": 0.5,
        "missing_as_attack": True,
        "n": int(n),
        "n_missing": int(missing.sum()),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "per_class_f1": metrics["per_class_f1"],
        "pos_precision": float(metrics["pos_precision"]),
        "pos_recall": float(metrics["pos_recall"]),
    }
    if (~missing).any():
        evaluable = classification_metrics(
            labels[~missing], predictions[~missing], ["安全", "攻击性"])
        metrics_payload["evaluable_only"] = {
            "n": int((~missing).sum()),
            "accuracy": float(evaluable["accuracy"]),
            "macro_f1": float(evaluable["macro_f1"]),
        }

    error_types: dict[str, int] = {}
    for _, row in frame[frame["status"].isin(["blocked", "error"])].iterrows():
        key = "|".join(
            str(value) for value in
            (row["error_code"], row["error_type"], row["error_message"])
            if pd.notna(value)
        )
        key = key[:160] or "unknown"
        error_types[key] = error_types.get(key, 0) + 1

    verdict = release_verdict(status_counts, n)
    return {
        "task": "D8_试点",
        "run": {
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": finished.isoformat(timespec="seconds"),
            "finished_at_beijing": finished.astimezone(BEIJING).isoformat(timespec="seconds"),
            "elapsed_sec": float((finished - started).total_seconds()),
            "workers": int(args.workers),
            "max_retries": int(args.max_retries),
            "peak_at_start": bool(is_peak(started)),
            "note": "本轮只做 200 条试点；未跑全量。",
        },
        "model": {
            "name": args.model,
            "version": "DeepSeek-V4.1-Flash-1017",
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "api_docs": "https://api-docs.deepseek.com/",
            "api_format": "OpenAI 兼容 Chat Completions",
            "thinking": "disabled",
            "request": {
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": MAX_TOP_LOGPROBS,
                "prompt": "P2",
                "prompt_text": PROMPTS["P2"],
                "system_prompt": SYSTEM_PROMPT,
            },
        },
        "sampling": meta,
        "n": int(n),
        "status_counts": status_counts,
        "status_ratios": status_ratios,
        "has_prob": {"count": int(has_prob), "ratio": has_prob / n,
                     "definition": "ok + partial"},
        "blocked_error": {"count": int(blocked_error),
                          "ratio": blocked_error / n,
                          "definition": "blocked + error"},
        "error_types": error_types,
        "avg_prompt_tokens": prompt_mean,
        "avg_completion_tokens": completion_mean,
        "avg_latency_ms": latency_mean,
        "avg_latency_ms_ok": latency_ok,
        "p95_latency_ms": latency_p95,
        "metrics_at_0.5": metrics_payload,
        "pricing": {
            "source_urls": [PRICING_SOURCE_CN, PRICING_SOURCE_EN],
            "fetched_at": "2026-11-22",
            "unit": "每百万 tokens",
            "currency_primary": "CNY",
            "raw_cn": PRICING_RAW_CN,
            "raw_en": PRICING_RAW_EN,
            "unit_prices_cny": PRICING_CN,
            "unit_prices_usd": PRICING_USD,
            "off_peak_note": OFF_PEAK_NOTE,
            "cache_assumption": (
                "主口径按缓存未命中计（评论互不重复）；另给输入全命中的下界。"),
            "token_basis": {
                "n_with_usage": int(frame["prompt_tokens"].notna().sum()),
                "prompt": "有 usage 的条目取平均",
                "completion": "有 usage 的条目取平均",
            },
            "cost_per_1k": costs,
        },
        "release_conditions": verdict,
    }


def print_summary(payload: dict) -> None:
    """终端输出：状态、比例、费用、放行条件。"""
    n = payload["n"]
    counts = payload["status_counts"]
    print("模型：{}（{}）".format(payload["model"]["name"],
                                  payload["model"]["endpoint"]))
    print("状态计数：" + "，".join(
        f"{key}={value}（{value / n:.1%}）" for key, value in counts.items()))
    print("有概率比例：{:.1%}（ok+partial={}）".format(
        payload["has_prob"]["ratio"], payload["has_prob"]["count"]))
    print("拦截与报错合计：{:.1%}（blocked+error={}）".format(
        payload["blocked_error"]["ratio"], payload["blocked_error"]["count"]))
    print("平均词元：prompt={}，completion={}".format(
        payload["avg_prompt_tokens"], payload["avg_completion_tokens"]))
    costs = payload["pricing"]["cost_per_1k"]
    print("每千条费用（CNY，缓存未命中）：谷时={}，标准价={}".format(
        costs.get("cost_per_1k_cny_off_peak_cache_miss"),
        costs.get("cost_per_1k_cny_peak_cache_miss")))
    print("平均延迟：{:.0f} ms".format(payload["avg_latency_ms"]))
    metrics = payload["metrics_at_0.5"]
    print("{} 条阈值 0.5：准确率={:.4f}，宏 F1={:.4f}".format(
        n, metrics["accuracy"], metrics["macro_f1"]))
    for item in payload["release_conditions"]["items"]:
        print("放行条件：{} -> {}（{}）".format(
            item["name"], "满足" if item["pass"] else "不满足",
            f"{item['ratio']:.2%}"))
    print("放行条件总体：{}".format(
        "满足" if payload["release_conditions"]["overall_pass"] else "不满足"))


def resolve_endpoint(base_url: str, endpoint: str = "") -> str:
    if endpoint:
        return endpoint.rstrip("/")
    return base_url.rstrip("/") + "/chat/completions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="D8 DeepSeek API 零样本试点")
    parser.add_argument("--pilot", action="store_true",
                        help="跑 200 条试点（本轮必须加）")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help="试点条数，默认 200；调试可改小")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--endpoint", default="",
                        help="完整接口地址；为空时由 base_url 拼 /chat/completions")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--flush_every", type=int, default=10,
                        help="每多少条写一次 parquet；0 表示只在结束时写")
    parser.add_argument("--out_dir", default=str(RESULTS))
    parser.add_argument("--name", default=DEFAULT_NAME,
                        help="结果名；文件名会加 D8_ 前缀")
    parser.add_argument("--log_dir", default=str(ROOT / "logs"))
    parser.add_argument("--peak_action", choices=("pause", "wait"), default="pause",
                        help="遇到高峰是退出续跑还是等到谷时")
    parser.add_argument("--dry_run", "--dry-run", dest="dry_run", action="store_true",
                        help="不调用 API、不写结果，只核对抽样与消息")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if not args.pilot:
        parser.error("本轮只做试点，请加 --pilot")
    if not 1 <= args.limit <= DEFAULT_LIMIT:
        parser.error(f"--limit 必须在 1～{DEFAULT_LIMIT}（本轮只做试点）")
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers 必须在 1～{MAX_WORKERS}")
    if not 0 <= args.max_retries <= MAX_RETRIES:
        parser.error(f"--max_retries 必须在 0～{MAX_RETRIES}")
    if args.timeout <= 0:
        parser.error("--timeout 必须为正数")
    if args.flush_every < 0:
        parser.error("--flush_every 不能为负")
    if not args.name or Path(args.name).name != args.name or args.name in (".", ".."):
        parser.error("--name 只能是单个文件名片段，不能带路径")
    if not args.model.strip():
        parser.error("--model 不能为空")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.endpoint = resolve_endpoint(args.base_url, args.endpoint)
    sample, meta = pilot_sample(args.limit, args.seed)
    out_dir = resolve_results_dir(args.out_dir)
    root = ROOT.resolve()
    if not out_dir.resolve().is_relative_to(root):
        parser.error("--out_dir 必须在本工作目录内")
    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = (ROOT / log_dir).resolve()
    if not log_dir.resolve().is_relative_to(root):
        parser.error("--log_dir 必须在本工作目录内")

    stem = f"D8_{args.name}"
    out_json = out_dir / f"{stem}.json"
    out_parquet = out_dir / "preds" / f"{stem}.parquet"
    if args.dry_run:
        first = sample.iloc[0]
        messages = build_messages("P2", clean_comment(first["text"]))
        print(json.dumps({
            "dry_run": True,
            "model": args.model,
            "base_url": args.base_url,
            "endpoint": args.endpoint,
            "n": int(len(sample)),
            "label_counts": meta["label_counts"],
            "first_index": int(first["index"]),
            "first_label": int(first["label"]),
            "message_roles": [message["role"] for message in messages],
            "message_lengths": [len(message["content"]) for message in messages],
            "request": {"temperature": 0, "max_tokens": 1,
                        "logprobs": True, "top_logprobs": MAX_TOP_LOGPROBS,
                        "thinking": {"type": "disabled"}},
        }, ensure_ascii=False, indent=2))
        return 0

    existing = load_results(out_parquet)
    done = {int(value) for value in existing["index"]} if len(existing) else set()
    pending = [row for row in sample.itertuples(index=False)
               if int(row.index) not in done]
    api_key = load_api_key() if pending else ""
    logger = make_logger(log_dir / f"{stem}.log", api_key)
    started = datetime.now().astimezone()
    logger.info("D8 试点开始：模型=%s 端点=%s 待跑=%d/%d 并发=%d",
                args.model, args.endpoint, len(pending), len(sample), args.workers)
    try:
        if pending:
            client = DeepSeekClient(
                api_key, model=args.model, endpoint=args.endpoint,
                timeout=args.timeout, max_retries=args.max_retries,
                logger=logger)
            frame = run_pending(
                client, sample, out_parquet, workers=args.workers,
                flush_every=args.flush_every, logger=logger,
                peak_check=make_peak_check(args.peak_action, logger))
        else:
            frame = existing
            logger.info("全部 index 已有结果，跳过 API 调用")
    except PeakPause as exc:
        logger.warning("%s", exc)
        print(f"已暂停：{exc}")
        return 3

    frame = frame[frame["index"].isin(sample["index"].tolist())].copy()
    missing = sorted(set(int(v) for v in sample["index"]) - set(int(v) for v in frame["index"]))
    if missing:
        logger.error("还有 %d 条没有结果，未写汇总 JSON", len(missing))
        print(f"还有 {len(missing)} 条没有结果，未写汇总 JSON")
        return 2
    finished = datetime.now().astimezone()
    payload = summarize(frame, meta, args, started, finished)
    save_json(out_json, payload)
    logger.info("试点汇总写入 %s（%d 条）", out_json, len(frame))
    print_summary(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





