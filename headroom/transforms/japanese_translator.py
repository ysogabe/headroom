"""Japanese → English translation preprocessing transform."""
from __future__ import annotations

import logging
import re
import threading
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from ..utils import deep_copy_messages
from .base import Transform, split_frozen

logger = logging.getLogger(__name__)

_JA_EN_MODEL_ID = "Helsinki-NLP/opus-mt-ja-en"

# CJK 検出 regex（estimator.py の CJK_PATTERN に準拠）
_CJK_PATTERN = re.compile(
    "[　-〿぀-ヿ㐀-䶿一-鿿"
    "가-힯豈-﫿＀-￯"
    "\U00020000-\U0002a6df]"
)

# モジュールレベルのシングルトン（モデルは1回だけロード）
_model_lock = threading.Lock()
_model = None
_tokenizer = None
_load_thread: threading.Thread | None = None
_load_failed = False


def _has_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def _background_load() -> None:
    global _model, _tokenizer, _load_failed
    try:
        logger.info("JapaneseTranslator: loading %s ...", _JA_EN_MODEL_ID)
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(_JA_EN_MODEL_ID)
        mdl = MarianMTModel.from_pretrained(_JA_EN_MODEL_ID)
        with _model_lock:
            _tokenizer = tok
            _model = mdl
        logger.info("JapaneseTranslator: model loaded")
    except Exception as exc:
        logger.warning("JapaneseTranslator: load failed: %s", exc)
        with _model_lock:
            _load_failed = True


def ensure_background_load() -> None:
    """バックグラウンドでモデルをロード開始（冪等、非ブロッキング）。"""
    global _load_thread
    with _model_lock:
        if _model is not None or _load_failed:
            return
        if _load_thread is not None and _load_thread.is_alive():
            return
        t = threading.Thread(target=_background_load,
                             name="japanese-translator-load", daemon=True)
        _load_thread = t
    t.start()


def _translate(text: str) -> str:
    """テキストを英語に翻訳。失敗時は原文をそのまま返す。"""
    with _model_lock:
        mdl, tok = _model, _tokenizer
    if mdl is None or tok is None:
        return text
    try:
        inputs = tok([text], return_tensors="pt", padding=True,
                     truncation=True, max_length=512)
        outputs = mdl.generate(**inputs)
        result = tok.decode(outputs[0], skip_special_tokens=True)
        return result if result.strip() else text
    except Exception as exc:
        logger.debug("JapaneseTranslator: translation error: %s", exc)
        return text


class JapaneseTranslationTransform(Transform):
    """CJK テキストを英語に翻訳してから KompressCompressor に渡す。

    翻訳対象: user / tool ロールのメッセージ（非フリーズ部分のみ）
    翻訳しない: system/developer（キャッシュ対象）、assistant（LLM 出力）
    """

    name = "japanese_translation"

    def __init__(self) -> None:
        ensure_background_load()

    def should_apply(self, messages, tokenizer: Tokenizer, **kwargs: Any) -> bool:
        with _model_lock:
            if _load_failed:
                return False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and _has_cjk(content):
                return True
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        t = block.get("text", "") or block.get("content", "")
                        if isinstance(t, str) and _has_cjk(t):
                            return True
        return False

    def apply(self, messages: list[dict[str, Any]], tokenizer: Tokenizer,
              **kwargs: Any) -> TransformResult:
        tokens_before = tokenizer.count_messages(messages)
        result = deep_copy_messages(messages)
        frozen_count = kwargs.get("frozen_message_count", 0)
        applied: list[str] = []

        _, mutable = split_frozen(result, frozen_count)

        for msg in mutable:
            role = msg.get("role", "")
            if role in {"system", "developer", "assistant"}:
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                if _has_cjk(content):
                    t = _translate(content)
                    if t != content:
                        msg["content"] = t
                        applied.append(f"japanese_translation:{role}")
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    for key in ("text", "content"):
                        text = block.get(key, "")
                        if isinstance(text, str) and _has_cjk(text):
                            t = _translate(text)
                            if t != text:
                                block[key] = t
                                applied.append(f"japanese_translation:{role}:block")
                            break

        tokens_after = tokenizer.count_messages(result)
        return TransformResult(
            messages=result,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=applied,
        )
