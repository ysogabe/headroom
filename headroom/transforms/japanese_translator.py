"""Japanese → English translation preprocessing transform.

Uses Qwen/Qwen2.5-1.5B-Instruct via HuggingFace pipeline.
On Apple Silicon (M3+) the model runs on MPS; falls back to CPU elsewhere.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeoutError
from typing import Any

from ..config import TransformResult
from ..tokenizer import Tokenizer
from ..utils import deep_copy_messages
from .base import Transform, split_frozen

logger = logging.getLogger(__name__)

_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM_PROMPT = (
    "You are a Japanese-to-English translator. "
    "Translate the Japanese text provided by the user into natural English. "
    "Output only the translation, with no explanations or extra text."
)

# CJK 検出 regex（estimator.py の CJK_PATTERN に準拠）
_CJK_PATTERN = re.compile(
    "[　-〿぀-ヿ㐀-䶿一-鿿"
    "가-힯豈-﫿＀-￯"
    "\U00020000-\U0002a6df]"
)

# モジュールレベルのシングルトン（モデルは1回だけロード）
_model_lock = threading.Lock()
_pipe = None  # transformers pipeline instance
_load_thread: threading.Thread | None = None
_load_failed = False

# デフォルト 30 秒（LLM 推論は MarianMT より時間がかかるため余裕を持たせる）
TRANSLATION_TIMEOUT_SECONDS: float = float(
    os.environ.get("HEADROOM_TRANSLATION_TIMEOUT_SECONDS", "30")
)

# max_workers=1: 逐次推論でメモリ圧力を抑える
_translation_executor = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="ja-translate"
)
_translation_leaked_threads: int = 0
_translation_leaked_lock = threading.Lock()


def _has_cjk(text: str) -> bool:
    return bool(_CJK_PATTERN.search(text))


def _background_load() -> None:
    global _pipe, _load_failed
    try:
        import torch
        # transformers 5.x uses _LazyModule for top-level exports, which is not
        # always thread-safe when accessed from a background thread while the
        # main thread is still initialising the package.  Importing directly
        # from the concrete sub-package bypasses the lazy-loading machinery.
        from transformers.pipelines import pipeline as hf_pipeline

        logger.info("JapaneseTranslator: loading %s ...", _MODEL_ID)

        if torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.bfloat16   # Apple Silicon ネイティブ対応
        else:
            device = "cpu"
            dtype = torch.float32

        pipe = hf_pipeline(
            "text-generation",
            model=_MODEL_ID,
            torch_dtype=dtype,
            device=device,
        )
        with _model_lock:
            _pipe = pipe
        logger.info("JapaneseTranslator: model loaded (device=%s)", device)
    except Exception as exc:
        logger.warning("JapaneseTranslator: load failed: %s", exc)
        with _model_lock:
            _load_failed = True


def ensure_background_load() -> None:
    """バックグラウンドでモデルをロード開始（冪等、非ブロッキング）。"""
    global _load_thread
    with _model_lock:
        if _pipe is not None or _load_failed:
            return
        if _load_thread is not None and _load_thread.is_alive():
            return
        t = threading.Thread(target=_background_load,
                             name="japanese-translator-load", daemon=True)
        _load_thread = t
    t.start()


def _translate_batch(texts: list[str]) -> list[str]:
    """複数テキストを逐次 LLM 推論で日→英翻訳する。

    タイムアウトまたは失敗時は原文リストをそのまま返す（fail-open）。
    MarianMT と異なり 512 トークン制限はなく、長文も処理できる。
    """
    with _model_lock:
        pipe = _pipe
    if pipe is None:
        return texts

    def _infer_all() -> list[str]:
        results = []
        for text in texts:
            messages = [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ]
            out = pipe(messages, max_new_tokens=512, do_sample=False)
            # pipeline の戻り値: [{"generated_text": [{role, content}, ...]}]
            translated = out[0]["generated_text"][-1]["content"].strip()
            results.append(translated if translated else text)
        return results

    future = _translation_executor.submit(_infer_all)
    try:
        results = future.result(timeout=TRANSLATION_TIMEOUT_SECONDS)
        return [r if r.strip() else t for r, t in zip(results, texts)]
    except _FuturesTimeoutError:
        global _translation_leaked_threads
        with _translation_leaked_lock:
            _translation_leaked_threads += 1
        logger.warning(
            "JapaneseTranslator: batch inference timed out after %.1fs",
            TRANSLATION_TIMEOUT_SECONDS,
        )
        return texts
    except Exception as exc:
        logger.debug("JapaneseTranslator: batch translation error: %s", exc)
        return texts


def get_translation_stats() -> dict:
    """翻訳エグゼキュータの統計を返す（/stats エンドポイント向け）。"""
    with _translation_leaked_lock:
        leaked = _translation_leaked_threads
    return {
        "timeout_seconds": TRANSLATION_TIMEOUT_SECONDS,
        "leaked_threads_total": leaked,
    }


class JapaneseTranslationTransform(Transform):
    """CJK テキストを英語に翻訳してから KompressCompressor に渡す。

    翻訳対象: user ロールのメッセージ（非フリーズ部分のみ）
    翻訳しない: system/developer（キャッシュ対象）、assistant/tool（LLM 出力）
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

        # 第 1 パス: 翻訳対象テキストと書き戻し先を収集
        locations: list[tuple[int, int | None, str | None, str]] = []
        texts: list[str] = []

        for msg_idx, msg in enumerate(mutable):
            role = msg.get("role", "")
            if role in {"system", "developer", "assistant", "tool"}:
                continue

            content = msg.get("content", "")
            if isinstance(content, str):
                if _has_cjk(content):
                    locations.append((msg_idx, None, None, role))
                    texts.append(content)
            elif isinstance(content, list):
                for blk_idx, block in enumerate(content):
                    if not isinstance(block, dict):
                        continue
                    for key in ("text", "content"):
                        text = block.get(key, "")
                        if isinstance(text, str) and _has_cjk(text):
                            locations.append((msg_idx, blk_idx, key, role))
                            texts.append(text)
                            break

        # バッチ翻訳（CJK テキストがある場合のみ）
        if texts:
            translated = _translate_batch(texts)

            # 第 2 パス: 翻訳結果を書き戻す
            for (msg_idx, blk_idx, key, role), orig, trans in zip(locations, texts, translated):
                if trans == orig:
                    continue
                if blk_idx is None:
                    mutable[msg_idx]["content"] = trans
                    applied.append(f"japanese_translation:{role}")
                else:
                    mutable[msg_idx]["content"][blk_idx][key] = trans
                    applied.append(f"japanese_translation:{role}:block")

        tokens_after = tokenizer.count_messages(result)
        return TransformResult(
            messages=result,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=applied,
        )
