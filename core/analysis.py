"""单条评论分析引擎（P2）：提示词、严格 schema、证据子串校验、规则兜底。

不依赖 UI，也不直接写库；批处理与落库在 core/batch.py。
"""
from __future__ import annotations

import json
import re

from core.llm import LLMBackend, LLMCallError

THEMES = [
    "account_login",
    "pricing_subscription",
    "answer_quality",
    "speed_latency",
    "stability_crash",
    "missing_features",
    "ui_usability",
    "privacy_security",
    "customer_support",
    "sync_platform",
    "model_limits",
    "other",
]
SENTIMENTS = {"positive", "neutral", "mixed", "negative"}
INTENTS = {"praise", "complaint", "feature_request", "question", "other"}
SCHEMA_FIELDS = (
    "topics",
    "sentiment",
    "intent",
    "pain_point",
    "feature_request",
    "severity",
    "evidence_quote",
    "confidence",
    "needs_human_review",
)

SYSTEM_PROMPT_V20 = """You are a product analytics assistant for mobile app reviews.
Return ONLY one JSON object (no markdown, no prose) with EXACTLY these keys:
{
  "topics": ["..."],
  "sentiment": "positive|neutral|mixed|negative",
  "intent": "praise|complaint|feature_request|question|other",
  "pain_point": "short phrase or null",
  "feature_request": "short phrase or null",
  "severity": 0,
  "evidence_quote": "exact contiguous substring copied from the review",
  "confidence": 0.0,
  "needs_human_review": false
}
Rules:
- topics: 1-3 values, each MUST be one of: {themes}
- severity: integer 0-5 (0 = no problem at all, 5 = blocks usage or causes churn)
- evidence_quote: copy verbatim from the review, same words and same order; never paraphrase or translate
- confidence: 0.0-1.0; if the review is too short or ambiguous, use < 0.5 and set needs_human_review=true
- pain_point / feature_request: <= 12 words, or null
"""

# v2.1-p2：补充主题边界、answer_quality 与 model_limits 的区分、pain_point 证据约束、
# severity 0-5 锚点，并禁止把评分标准原文抄进 pain_point / feature_request。
SYSTEM_PROMPT_V21 = """You are a product analytics assistant for mobile app reviews of AI assistant apps.
Return ONLY one JSON object (no markdown, no prose) with EXACTLY these keys:
{
  "topics": ["..."],
  "sentiment": "positive|neutral|mixed|negative",
  "intent": "praise|complaint|feature_request|question|other",
  "pain_point": "short phrase or null",
  "feature_request": "short phrase or null",
  "severity": 0,
  "evidence_quote": "exact contiguous substring copied from the review",
  "confidence": 0.0,
  "needs_human_review": false
}
Core rules:
- topics: 1-3 values from the theme list below. Only choose a theme when the review text itself contains concrete evidence for it. Do not infer a theme from the star rating alone. If no theme has evidence, use "other".
- Do not list several themes just in case; use the smallest set that the text supports.
- evidence_quote: copy a contiguous span verbatim from the review, same words and same order; never paraphrase or translate.
- pain_point: a short phrase (<= 12 words) naming the specific problem the user actually states. It MUST be directly supported by evidence_quote. If the review only gives a general judgement (for example "worst app", "good app", "very nice") without a concrete issue, return null. Never invent a specific failure such as crashes, data loss or billing errors from the star rating or from an overall impression.
- feature_request: <= 12 words, a capability the user explicitly asks for, or null.
- severity: integer 0-5, judged from the concrete issue described in the review, NOT from the star rating. Use all levels, including 1:
  0 = no specific problem: pure praise, vague comment, or no usable information.
  1 = minor inconvenience: occasional or easy to work around; the main task is still easy to complete.
  2 = a secondary feature is affected or noticeably slower; the main task is still doable.
  3 = a main use case is clearly impaired; the user needs a workaround.
  4 = core experience severely affected: a frequent core flow is largely unusable or needs major extra cost.
  5 = the app is unusable for the task, or money, privacy or account trust is harmed.
  Do not round everything up to 5 or down to 0; if the issue is real but small, choose 1.
- confidence: 0.0-1.0; if the review is too short or ambiguous, use < 0.5 and set needs_human_review=true.
- needs_human_review: true when the theme or severity is uncertain, or the quote is imperfect.
- Do not copy this rubric's wording into pain_point or feature_request; use the user's own words.

Theme definitions and boundaries (all codes): {themes}
- account_login: sign-in, sign-up, account access, verification codes, login failures, account deletion or bans.
- pricing_subscription: price, free tier, subscription billing, paywalls, refunds, cancellation, unwanted charges.
- answer_quality: the generated answer itself is wrong, inaccurate, hallucinated, off-topic or does not answer the question. Use this whenever the correctness or relevance of the answer is criticised.
- speed_latency: slow responses, long waits, timeouts, lag between question and answer.
- stability_crash: the app crashes, freezes, closes unexpectedly, fails to launch, or repeated errors block use.
- missing_features: a specific capability the user wants is absent (for example voice mode, file upload, export, dark mode).
- ui_usability: layout, navigation, confusing controls, readability, ads placement, difficulty finding features.
- privacy_security: data collection, permissions, data leakage, safety of conversations or personal information.
- customer_support: support channels, response time, refund help, unresolved tickets.
- sync_platform: cross-device sync, web/mobile consistency, history transfer, platform availability.
- model_limits: inherent capability limits or coverage gaps of the underlying model, such as context length, no internet browsing, cannot read files, limited languages, weak reasoning. Use this when "the model cannot do X"; do NOT use it when the answer is simply wrong (that is answer_quality).
- other: only when no theme above has concrete evidence in the text, e.g. generic praise or complaint such as "good app", "very nice", or emoji-only reviews.
"""

# v2.2-p2：只收紧 missing_features 边界，其余主题定义、字段、校验规则与 v2.1 完全一致。
MISSING_FEATURES_BOUNDARY_V22 = """

MISSING_FEATURES boundary (highest priority; these rules override the one-line definition above):

Decision order for "missing_features":
1. REQUIRED: the review must explicitly signal absence or request for a concrete capability, using language such as "no / none / cannot / can't / doesn't have / not available / not supported / missing / lacks / removed / bring back / please add / wish / hope / when will ... be available". Name the specific capability the user says is absent (e.g. voice mode, file upload, export, dark mode).
2. If the review praises or merely describes an existing feature -> NOT missing_features; use "other" unless another theme has evidence.
3. If an existing feature is broken, unreliable or fails intermittently ("spotty", "doesn't save", "won't work") -> "stability_crash" when it blocks use, or "answer_quality" when only the output is wrong; NOT missing_features.
4. If the answer is off-topic or does not answer the question -> "answer_quality"; NOT missing_features.
5. If access is blocked by policy, region, paywall or usage quota -> "model_limits" or "pricing_subscription"; NOT missing_features.
6. If the review only says "too restrictive / too limited / not enough features" without naming what is missing -> "model_limits" or "other"; NOT missing_features.
7. If the user explicitly says a capability was removed or asks to restore it ("bring back", "why did you remove") -> missing_features IS correct.

Conflict check:
- One complaint must receive only one theme among missing_features / answer_quality / model_limits / stability_crash. Different complaints in the same review may receive different themes.
- Never tag the same sentence both missing_features and answer_quality, or both missing_features and model_limits.
- When unsure between missing_features and other, choose "other" and set needs_human_review=true.
"""

SYSTEM_PROMPT_V22 = SYSTEM_PROMPT_V21 + MISSING_FEATURES_BOUNDARY_V22

# v2.3：只修订四条通用判定口径（主题优先序、severity 锚点、意图顺序、情感 neutral），
# 其余字段、主题定义与 v2.2 完全一致；用于 60 条正式标注样本的对比验证。
DECISION_RULES_V23 = """

v2.3 DECISION ORDER (these rules override any conflicting guidance above):

1) THEME ORDER
- answer_quality takes precedence over model_limits. If the user criticises the answer itself (wrong, inaccurate, hallucinated, outdated, off-topic, or does not answer the question), use answer_quality, even when the cause looks like a model capability.
- Use model_limits ONLY for inherent capability boundaries: context length, no internet browsing, cannot read files / images / audio, unsupported language, no memory, policy refusal. Do NOT use it when the answer is simply wrong or low quality.
- If at least one concrete theme has textual evidence, NEVER add "other". "other" is allowed only when no concrete theme has evidence.
- Keep the smallest theme set supported by the text; do not tag several overlapping themes for the same complaint.

2) SEVERITY ANCHORS (judge the concrete issue, never the tone)
- 0 = pure emotion or venting with no concrete failure, generic praise, or an uninformative comment.
- 2 = a clear impact on a secondary flow or noticeable degradation, while the main task is still doable.
- 3 = a main use case is clearly impaired and needs a workaround.
- 4 = a frequent core flow is largely unusable; 5 = the app is unusable for the task, or money, privacy or account trust is harmed.
- Never assign 3-5 merely because of profanity, exclamation marks, ALL CAPS or an angry tone; tone affects sentiment, not severity.
- If the text is too short or ambiguous to identify a concrete issue, set "severity": null and "needs_human_review": true instead of guessing a level.

3) INTENT ORDER
- First check feature_request: the user asks for a capability, including "I hope / wish / please add / could you add / would be nice / when will X be available / bring back".
- Then check question: the user asks an actual question ("why", "how do I", "is there", "what happened to").
- Then complaint: the user reports a problem or dissatisfaction without asking for a capability or asking a question.
- Then praise for positive evaluation; other for neutral statements or unclear intent.
- Never default to complaint.

4) SENTIMENT
- Pure factual statements, feature descriptions, comparisons without evaluation, or neutral observations -> neutral.
- Do not treat a short, terse or emotionally flat review as negative.
- neutral is a valid and common label; do not omit it in favour of positive or negative.
"""

SYSTEM_PROMPT_V23 = SYSTEM_PROMPT_V22 + DECISION_RULES_V23

STRICT_APPENDIX = (
    "\nYour previous reply was not valid JSON. Return ONLY the JSON object, "
    "no markdown, no explanation.\n"
)


def normalize_ws(s: str) -> str:
    """规范化空白：所有连续空白折叠为一个空格并去首尾（不做大小写改写）。"""
    return re.sub(r"\s+", " ", s or "").strip()


def _has_letters(s: str) -> bool:
    return len(re.findall(r"[A-Za-z]", s)) >= 3


def build_system_prompt(cfg: dict) -> str:
    """按 prompt_version 选择对应提示词；v2.0 文本保留以便复现与对照。"""
    themes = cfg.get("theme_codes") or THEMES
    version = str(cfg.get("prompt_version") or "")
    if version.startswith("v2.3"):
        base = SYSTEM_PROMPT_V23
    elif version.startswith("v2.2"):
        base = SYSTEM_PROMPT_V22
    elif version.startswith("v2.1"):
        base = SYSTEM_PROMPT_V21
    else:
        base = SYSTEM_PROMPT_V20
    return base.replace("{themes}", ", ".join(themes))


def build_user_prompt(app: str, score, content: str) -> str:
    return f'Review (app: {app}, star rating: {score}):\n"""{content}"""'


def _sentiment_from_score(score) -> str:
    try:
        s = int(score)
    except (TypeError, ValueError):
        return "neutral"
    if s <= 2:
        return "negative"
    if s >= 4:
        return "positive"
    return "mixed"


def _coerce_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def evaluate(raw_text: str, content: str, score, cfg: dict) -> dict:
    """解析 + 严格校验 + 自动补齐。返回结构统一供批处理落库与统计使用。"""
    analysis_cfg = cfg.get("analysis", {})
    min_quote_len = int(analysis_cfg.get("min_quote_len", 4))
    confidence_threshold = float(analysis_cfg.get("confidence_threshold", 0.5))
    themes_allowed = set(cfg.get("theme_codes") or THEMES)

    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {
            "parse_ok": False,
            "status": "failed",
            "error": f"json_parse:{e}",
            "raw_parsed": {},
            "themes": [],
            "sentiment": None,
            "intent": None,
            "pain_point": None,
            "feature_request": None,
            "severity": None,
            "evidence_quote": None,
            "confidence": None,
            "needs_human_review": 1,
            "is_valid": 0,
            "quote_ok": False,
            "missing_fields": [],
            "invalid_fields": [],
            "filled_fields": [],
            "flags": [],
        }
    if not isinstance(obj, dict):
        return {
            "parse_ok": False,
            "status": "failed",
            "error": "json_not_object",
            "raw_parsed": {},
            "themes": [],
            "sentiment": None,
            "intent": None,
            "pain_point": None,
            "feature_request": None,
            "severity": None,
            "evidence_quote": None,
            "confidence": None,
            "needs_human_review": 1,
            "is_valid": 0,
            "quote_ok": False,
            "missing_fields": [],
            "invalid_fields": [],
            "filled_fields": [],
            "flags": [],
        }

    errors: list[str] = []
    missing: list[str] = []
    invalid: list[str] = []
    filled: list[str] = []
    flags: list[str] = []

    # topics（兼容模型偶尔输出 themes / topic 单数）
    raw_topics = obj.get("topics", obj.get("themes", obj.get("topic", "__absent__")))
    if raw_topics == "__absent__":
        missing.append("topics")
        filled.append("topics")
        raw_topics = []
    if isinstance(raw_topics, str):
        raw_topics = [raw_topics]
    if not isinstance(raw_topics, list):
        invalid.append("topics")
        raw_topics = []
    topics: list[str] = []
    for t in raw_topics:
        if not isinstance(t, str):
            continue
        key = t.strip().lower().replace("-", "_").replace(" ", "_")
        if key in themes_allowed:
            topics.append(key)
    if not topics:
        topics = ["other"]
        if raw_topics:
            filled.append("topics_other")
            errors.append("topics_all_invalid")
    topics = list(dict.fromkeys(topics))

    # sentiment
    sentiment = obj.get("sentiment")
    if "sentiment" not in obj:
        missing.append("sentiment")
    if sentiment not in SENTIMENTS:
        if "sentiment" in obj:
            invalid.append("sentiment")
        sentiment = _sentiment_from_score(score)
        filled.append("sentiment")

    # intent
    intent = obj.get("intent")
    if "intent" not in obj:
        missing.append("intent")
    if intent not in INTENTS:
        if "intent" in obj:
            invalid.append("intent")
        intent = "other"
        filled.append("intent")

    # pain_point / feature_request：可选，允许 null；模型偶发输出字符串 "null" 等占位值，统一归一为 None
    _NULL_WORDS = {"null", "none", "n/a", "no specific problem", "no specific issue"}

    pain_point = obj.get("pain_point")
    if pain_point is not None and not isinstance(pain_point, str):
        invalid.append("pain_point")
        pain_point = None
    if isinstance(pain_point, str):
        pain_point = pain_point.strip() or None
        if pain_point and pain_point.lower() in _NULL_WORDS:
            pain_point = None

    feature_request = obj.get("feature_request")
    if feature_request is not None and not isinstance(feature_request, str):
        invalid.append("feature_request")
        feature_request = None
    if isinstance(feature_request, str):
        feature_request = feature_request.strip() or None
        if feature_request and feature_request.lower() in _NULL_WORDS:
            feature_request = None

    # severity：0-5 整数；缺失/非法则置空并转人工（不编造严重度）
    severity = obj.get("severity")
    if "severity" not in obj:
        missing.append("severity")
    if severity is not None:
        try:
            if isinstance(severity, bool):
                raise ValueError
            sev_f = float(severity)
            if sev_f != int(sev_f) or not 0 <= sev_f <= 5:
                raise ValueError
            severity = int(sev_f)
        except (TypeError, ValueError):
            invalid.append("severity")
            errors.append("severity_invalid")
            severity = None
    if severity is None and "severity" in obj and "severity" not in invalid:
        errors.append("severity_null")
    if severity is None:
        filled.append("severity_null")

    # confidence
    confidence = obj.get("confidence")
    if "confidence" not in obj:
        missing.append("confidence")
    if confidence is None:
        confidence = confidence_threshold
        filled.append("confidence")
    else:
        try:
            confidence = float(confidence)
            if confidence != confidence:  # NaN
                raise ValueError
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            invalid.append("confidence")
            confidence = confidence_threshold
            filled.append("confidence")

    # needs_human_review
    if "needs_human_review" not in obj:
        missing.append("needs_human_review")
    model_flags_review = _coerce_bool(obj.get("needs_human_review", False))

    # evidence_quote：规范化空白后必须是原文连续子串
    quote = obj.get("evidence_quote")
    quote_ok = False
    if "evidence_quote" not in obj:
        missing.append("evidence_quote")
    if not isinstance(quote, str) or not quote.strip():
        if "evidence_quote" in obj:
            invalid.append("evidence_quote")
        errors.append("quote_missing")
        quote = None
    else:
        quote = normalize_ws(quote.strip().strip('"').strip("“”").strip())
        if len(quote) < min_quote_len:
            errors.append("quote_too_short")
        elif quote not in normalize_ws(content):
            errors.append("quote_not_substring")
        else:
            quote_ok = True

    # 星级 vs 情感冲突打标
    try:
        star = int(score) if score is not None else None
    except (TypeError, ValueError):
        star = None
    if star is not None:
        if star <= 2 and sentiment == "positive":
            flags.append(f"star_sentiment_conflict(score={star},sentiment=positive)")
        if star >= 4 and sentiment == "negative":
            flags.append(f"star_sentiment_conflict(score={star},sentiment=negative)")
        if star <= 2 and intent == "praise":
            flags.append(f"star_intent_conflict(score={star},intent=praise)")
    if severity == 0 and sentiment == "negative":
        flags.append("severity_zero_with_negative")

    needs_human_review = (
        model_flags_review
        or confidence < confidence_threshold
        or not quote_ok
        or bool(flags)
        or bool(missing)
        or bool(invalid)
    )

    degraded = bool(missing) or bool(invalid) or bool(filled) or not quote_ok or severity is None
    status = "degraded" if degraded else "done"
    is_valid = 1 if status == "done" else 0

    error_parts = errors + [f"missing:{m}" for m in missing] + [f"invalid:{i}" for i in invalid]
    error_parts += [f"filled:{f}" for f in filled] + [f"flag:{f}" for f in flags]

    return {
        "parse_ok": True,
        "status": status,
        "error": "; ".join(error_parts),
        "raw_parsed": obj,
        "themes": topics,
        "sentiment": sentiment,
        "intent": intent,
        "pain_point": pain_point,
        "feature_request": feature_request,
        "severity": severity,
        "evidence_quote": quote,
        "confidence": confidence,
        "needs_human_review": 1 if needs_human_review else 0,
        "is_valid": is_valid,
        "quote_ok": quote_ok,
        "missing_fields": missing,
        "invalid_fields": invalid,
        "filled_fields": filled,
        "flags": flags,
    }


def _failure(error: str) -> dict:
    return {
        "parse_ok": False,
        "status": "failed",
        "error": error,
        "raw_parsed": {},
        "themes": [],
        "sentiment": None,
        "intent": None,
        "pain_point": None,
        "feature_request": None,
        "severity": None,
        "evidence_quote": None,
        "confidence": None,
        "needs_human_review": 1,
        "is_valid": 0,
        "quote_ok": False,
        "missing_fields": [],
        "invalid_fields": [],
        "filled_fields": [],
        "flags": [],
    }


def analyze_one(backend: LLMBackend, item: dict, cfg: dict) -> dict:
    """分析单条评论。规则可直接跳过的，不调用模型。"""
    content = item.get("content") or ""
    score = item.get("score")
    norm = normalize_ws(content)
    min_len = int(cfg.get("analysis", {}).get("min_content_len", 10))
    if len(norm) < min_len or not _has_letters(norm):
        result = _failure("rule_skip:too_short_or_meaningless")
        result["status"] = "skipped"
        result["seconds"] = 0.0
        result["attempts"] = 0
        result["raw_output"] = ""
        return result

    system = build_system_prompt(cfg)
    user = build_user_prompt(item.get("app", ""), score, content)
    total_seconds = 0.0
    last_raw = ""
    attempts = 0
    result = _failure("json_parse:no_attempt")

    for _ in range(2):
        try:
            resp = backend.chat_json(system, user)
        except LLMCallError as e:
            result = _failure(f"call:{e.code}")
            result["seconds"] = round(total_seconds, 2)
            result["attempts"] = attempts
            result["raw_output"] = last_raw
            return result
        except Exception as e:  # 单条失败隔离
            result = _failure(f"unexpected:{type(e).__name__}")
            result["seconds"] = round(total_seconds, 2)
            result["attempts"] = attempts
            result["raw_output"] = last_raw
            return result

        attempts += 1
        total_seconds += resp.seconds
        last_raw = resp.text
        result = evaluate(resp.text, content, score, cfg)
        if result["parse_ok"]:
            break
        system = system + STRICT_APPENDIX

    result["seconds"] = round(total_seconds, 2)
    result["attempts"] = attempts
    result["raw_output"] = last_raw
    return result
