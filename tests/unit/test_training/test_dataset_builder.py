"""Unit tests for DatasetBuilder chat-template formatting."""

from src.training.dataset_builder import DatasetBuilder, DEFAULT_SYSTEM_PROMPT


def _builder_without_init() -> DatasetBuilder:
    # Bypass __init__ (which needs a DB session + may download a tokenizer).
    b = DatasetBuilder.__new__(DatasetBuilder)
    b._tokenizer = None
    return b


def test_fallback_chat_template_used_when_no_tokenizer():
    b = _builder_without_init()
    text = b._format_example("What is 2+2?", "4")
    # Fallback should produce Llama-3 chat markers, NOT Alpaca "### Instruction".
    assert "### Instruction" not in text
    assert "<|start_header_id|>user<|end_header_id|>" in text
    assert "<|start_header_id|>assistant<|end_header_id|>" in text
    assert "What is 2+2?" in text
    assert "4" in text
    assert DEFAULT_SYSTEM_PROMPT in text


def test_tokenizer_chat_template_is_preferred():
    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            assert tokenize is False
            roles = [m["role"] for m in messages]
            assert roles == ["system", "user", "assistant"]
            return "TEMPLATED:" + messages[1]["content"] + "|" + messages[2]["content"]

    b = _builder_without_init()
    b._tokenizer = FakeTokenizer()
    out = b._format_example("hi", "hello")
    assert out == "TEMPLATED:hi|hello"
