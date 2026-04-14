"""Tests for llm_client.py — strip_thinking and RoleAwareLLMClient."""

import pytest
from llm_client import _strip_thinking, RoleAwareLLMClient, BaseLLMClient


class TestStripThinking:
    def test_removes_think_block(self):
        text = "<think>Some reasoning here.</think>Final answer."
        assert _strip_thinking(text) == "Final answer."

    def test_no_think_block(self):
        text = "Just a normal response."
        assert _strip_thinking(text) == "Just a normal response."

    def test_multiline_think(self):
        text = "<think>\nLine 1\nLine 2\n</think>\nAnswer"
        assert _strip_thinking(text) == "Answer"

    def test_empty_think(self):
        text = "<think></think>Result"
        assert _strip_thinking(text) == "Result"


class RecordingClient(BaseLLMClient):
    """Records the last call arguments for testing the wrapper."""

    def __init__(self):
        self.last_call = {}

    def generate(self, prompt, model=None, temperature=0.0,
                 max_tokens=4096, thinking=None):
        self.last_call = {
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking": thinking,
        }
        return "mock response"


class TestRoleAwareLLMClient:
    def test_align_role_enables_thinking(self):
        inner = RecordingClient()
        client = RoleAwareLLMClient(inner)
        client.generate("test prompt", model="Qwen/Qwen3-8B:align")
        assert inner.last_call["thinking"] is True
        assert inner.last_call["model"] == "Qwen/Qwen3-8B"

    def test_generate_role_disables_thinking(self):
        inner = RecordingClient()
        client = RoleAwareLLMClient(inner)
        client.generate("test prompt", model="Qwen/Qwen3-8B:generate")
        assert inner.last_call["thinking"] is False
        assert inner.last_call["model"] == "Qwen/Qwen3-8B"

    def test_no_role_defaults_thinking_false(self):
        inner = RecordingClient()
        client = RoleAwareLLMClient(inner)
        client.generate("test prompt", model="some-model")
        assert inner.last_call["thinking"] is False

    def test_explicit_thinking_overrides_role(self):
        inner = RecordingClient()
        client = RoleAwareLLMClient(inner)
        client.generate("test", model="model:generate", thinking=True)
        assert inner.last_call["thinking"] is True

    def test_none_model(self):
        inner = RecordingClient()
        client = RoleAwareLLMClient(inner)
        client.generate("test", model=None)
        assert inner.last_call["model"] is None
        assert inner.last_call["thinking"] is False
