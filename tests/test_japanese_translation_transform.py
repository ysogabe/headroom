"""Tests for JapaneseTranslationTransform.

Covers:
- _has_cjk() function
- should_apply() method
- apply() method (roles, frozen messages, list content blocks, TransformResult fields)
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as _FuturesTimeoutError
from unittest.mock import MagicMock, patch

import pytest

import headroom.transforms.japanese_translator as jt
from headroom.transforms.japanese_translator import (
    JapaneseTranslationTransform,
    _has_cjk,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transform() -> JapaneseTranslationTransform:
    """Create an instance without triggering the background model load."""
    return JapaneseTranslationTransform.__new__(JapaneseTranslationTransform)


def _make_tokenizer(count: int = 10) -> MagicMock:
    tok = MagicMock()
    tok.count_messages = MagicMock(return_value=count)
    return tok


# ---------------------------------------------------------------------------
# _has_cjk()
# ---------------------------------------------------------------------------

class TestHasCjk:
    def test_hiragana_returns_true(self):
        assert _has_cjk("こんにちは") is True

    def test_kanji_returns_true(self):
        assert _has_cjk("日本語") is True

    def test_ascii_only_returns_false(self):
        assert _has_cjk("hello") is False

    def test_empty_string_returns_false(self):
        assert _has_cjk("") is False

    def test_mixed_ascii_and_cjk_returns_true(self):
        assert _has_cjk("Hello World日本語") is True


# ---------------------------------------------------------------------------
# should_apply()
# ---------------------------------------------------------------------------

class TestShouldApply:
    def test_no_cjk_returns_false(self):
        transform = _make_transform()
        messages = [{"role": "user", "content": "Hello, how are you?"}]
        tokenizer = _make_tokenizer()
        assert transform.should_apply(messages, tokenizer) is False

    def test_load_failed_returns_false(self):
        transform = _make_transform()
        messages = [{"role": "user", "content": "日本語のテキスト"}]
        tokenizer = _make_tokenizer()
        with patch.object(jt, "_load_failed", True):
            result = transform.should_apply(messages, tokenizer)
        assert result is False

    def test_cjk_string_content_returns_true(self):
        transform = _make_transform()
        messages = [{"role": "user", "content": "日本語のテキスト"}]
        tokenizer = _make_tokenizer()
        with patch.object(jt, "_load_failed", False):
            result = transform.should_apply(messages, tokenizer)
        assert result is True

    def test_cjk_list_content_text_key_returns_true(self):
        transform = _make_transform()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "日本語"}],
            }
        ]
        tokenizer = _make_tokenizer()
        with patch.object(jt, "_load_failed", False):
            result = transform.should_apply(messages, tokenizer)
        assert result is True

    def test_cjk_list_content_content_key_returns_true(self):
        transform = _make_transform()
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "content": "日本語"}],
            }
        ]
        tokenizer = _make_tokenizer()
        with patch.object(jt, "_load_failed", False):
            result = transform.should_apply(messages, tokenizer)
        assert result is True


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

class TestApply:
    """Tests for the apply() method."""

    def _apply_with_mock_translate(
        self,
        messages: list[dict],
        translate_return: str = "translated text",
        frozen_message_count: int = 0,
        token_count: int = 10,
    ):
        transform = _make_transform()
        tokenizer = _make_tokenizer(token_count)
        # _translate_batch は texts: list[str] -> list[str] を返す
        def _batch_fn(texts):
            return [translate_return for _ in texts]
        with patch.object(jt, "_translate_batch", side_effect=_batch_fn) as mock_tr:
            result = transform.apply(
                messages, tokenizer, frozen_message_count=frozen_message_count
            )
        return result, mock_tr

    # ---- role skipping ----

    def test_system_role_is_skipped(self):
        messages = [{"role": "system", "content": "日本語のシステムプロンプト"}]
        result, mock_tr = self._apply_with_mock_translate(messages)
        mock_tr.assert_not_called()
        assert result.messages[0]["content"] == "日本語のシステムプロンプト"
        assert result.transforms_applied == []

    def test_assistant_role_is_skipped(self):
        messages = [{"role": "assistant", "content": "日本語のアシスタント応答"}]
        result, mock_tr = self._apply_with_mock_translate(messages)
        mock_tr.assert_not_called()
        assert result.messages[0]["content"] == "日本語のアシスタント応答"
        assert result.transforms_applied == []

    def test_user_role_is_translated(self):
        messages = [{"role": "user", "content": "日本語のメッセージ"}]
        result, mock_tr = self._apply_with_mock_translate(
            messages, translate_return="Japanese message"
        )
        mock_tr.assert_called_once_with(["日本語のメッセージ"])
        assert result.messages[0]["content"] == "Japanese message"
        assert "japanese_translation:user" in result.transforms_applied

    # ---- frozen messages ----

    def test_frozen_message_is_skipped(self):
        messages = [
            {"role": "user", "content": "日本語（フリーズ）"},
            {"role": "user", "content": "日本語（ミュータブル）"},
        ]
        result, mock_tr = self._apply_with_mock_translate(
            messages,
            translate_return="mutable translated",
            frozen_message_count=1,
        )
        # First message (frozen) must not be changed
        assert result.messages[0]["content"] == "日本語（フリーズ）"
        # Second message (mutable) should be translated
        assert result.messages[1]["content"] == "mutable translated"
        assert "japanese_translation:user" in result.transforms_applied

    # ---- no CJK ----

    def test_no_cjk_produces_empty_transforms_applied(self):
        messages = [{"role": "user", "content": "Hello, world!"}]
        result, mock_tr = self._apply_with_mock_translate(messages)
        mock_tr.assert_not_called()
        assert result.transforms_applied == []

    # ---- list content blocks ----

    def test_list_content_text_key_translated(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "日本語のブロック"}],
            }
        ]
        result, mock_tr = self._apply_with_mock_translate(
            messages, translate_return="block translated"
        )
        mock_tr.assert_called_once_with(["日本語のブロック"])
        assert result.messages[0]["content"][0]["text"] == "block translated"
        assert "japanese_translation:user:block" in result.transforms_applied

    def test_list_content_content_key_translated(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "content": "日本語のコンテント"}],
            }
        ]
        result, mock_tr = self._apply_with_mock_translate(
            messages, translate_return="content key translated"
        )
        mock_tr.assert_called_once_with(["日本語のコンテント"])
        assert result.messages[0]["content"][0]["content"] == "content key translated"
        assert "japanese_translation:user:block" in result.transforms_applied

    def test_list_content_non_cjk_not_translated(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "English only"}],
            }
        ]
        result, mock_tr = self._apply_with_mock_translate(messages)
        mock_tr.assert_not_called()
        assert result.transforms_applied == []

    # ---- TransformResult fields ----

    def test_transform_result_has_required_fields(self):
        messages = [{"role": "user", "content": "日本語"}]
        result, _ = self._apply_with_mock_translate(
            messages, translate_return="Japanese", token_count=42
        )
        assert result.tokens_before == 42
        assert result.tokens_after == 42
        assert isinstance(result.transforms_applied, list)
        assert isinstance(result.messages, list)

    def test_apply_does_not_mutate_original_messages(self):
        """apply() must deep-copy messages and leave originals intact."""
        original_content = "日本語のメッセージ"
        messages = [{"role": "user", "content": original_content}]
        result, _ = self._apply_with_mock_translate(
            messages, translate_return="translated"
        )
        # Original list is unchanged
        assert messages[0]["content"] == original_content

    def test_apply_with_mixed_roles(self):
        """Only user messages get translated; system and assistant are skipped."""
        messages = [
            {"role": "system", "content": "日本語のシステム"},
            {"role": "user", "content": "日本語のユーザー"},
            {"role": "assistant", "content": "日本語のアシスタント"},
        ]
        result, mock_tr = self._apply_with_mock_translate(
            messages, translate_return="en"
        )
        assert result.messages[0]["content"] == "日本語のシステム"
        assert result.messages[1]["content"] == "en"
        assert result.messages[2]["content"] == "日本語のアシスタント"
        assert result.transforms_applied == ["japanese_translation:user"]


# ---------------------------------------------------------------------------
# Timeout / batch behavior
# ---------------------------------------------------------------------------

class TestTranslateTimeout:
    """Tests for timeout and batch behavior."""

    def test_translate_timeout_returns_original(self):
        """future.result がタイムアウトすると原文リストをそのまま返す。"""
        import headroom.transforms.japanese_translator as jt_mod
        mock_pipe = MagicMock()

        with patch.object(jt_mod, "_pipe", mock_pipe):
            mock_future = MagicMock()
            mock_future.result.side_effect = _FuturesTimeoutError()
            with patch.object(jt_mod._translation_executor, "submit", return_value=mock_future):
                result = jt_mod._translate_batch(["日本語テスト"])

        assert result == ["日本語テスト"]

    def test_translate_timeout_increments_leaked_counter(self):
        """タイムアウト時に _translation_leaked_threads が +1 される。"""
        import headroom.transforms.japanese_translator as jt_mod

        before = jt_mod._translation_leaked_threads
        mock_pipe = MagicMock()

        with patch.object(jt_mod, "_pipe", mock_pipe):
            mock_future = MagicMock()
            mock_future.result.side_effect = _FuturesTimeoutError()
            with patch.object(jt_mod._translation_executor, "submit", return_value=mock_future):
                jt_mod._translate_batch(["日本語テスト"])

        assert jt_mod._translation_leaked_threads == before + 1

    def test_apply_uses_batch_translation(self):
        """apply() が複数 CJK メッセージを _translate_batch で 1 回だけ処理する。"""
        from headroom.transforms.japanese_translator import JapaneseTranslationTransform
        import headroom.transforms.japanese_translator as jt_mod

        transform = JapaneseTranslationTransform.__new__(JapaneseTranslationTransform)
        tok = MagicMock()
        tok.count_messages = MagicMock(return_value=10)

        messages = [
            {"role": "user", "content": "日本語その一"},
            {"role": "user", "content": "日本語その二"},
        ]

        def _batch_fn(texts):
            return ["translated" for _ in texts]

        with patch.object(jt_mod, "_translate_batch", side_effect=_batch_fn) as mock_tr:
            result = transform.apply(messages, tok)

        # _translate_batch は 1 回だけ呼ばれる
        mock_tr.assert_called_once()
        # 呼ばれた引数はテキストリスト（両方含む）
        call_args = mock_tr.call_args[0][0]
        assert "日本語その一" in call_args
        assert "日本語その二" in call_args

    def test_apply_single_message_calls_batch_once(self):
        """1 件の CJK メッセージでも _translate_batch 経由（1 回呼ばれる）。"""
        from headroom.transforms.japanese_translator import JapaneseTranslationTransform
        import headroom.transforms.japanese_translator as jt_mod

        transform = JapaneseTranslationTransform.__new__(JapaneseTranslationTransform)
        tok = MagicMock()
        tok.count_messages = MagicMock(return_value=10)

        messages = [{"role": "user", "content": "日本語"}]

        def _batch_fn(texts):
            return ["Japanese" for _ in texts]

        with patch.object(jt_mod, "_translate_batch", side_effect=_batch_fn) as mock_tr:
            result = transform.apply(messages, tok)

        mock_tr.assert_called_once()

    def test_get_translation_stats_structure(self):
        """get_translation_stats() が正しいキーを返す。"""
        from headroom.transforms.japanese_translator import get_translation_stats
        stats = get_translation_stats()
        assert "timeout_seconds" in stats
        assert "leaked_threads_total" in stats
        assert isinstance(stats["timeout_seconds"], float)
        assert isinstance(stats["leaked_threads_total"], int)
