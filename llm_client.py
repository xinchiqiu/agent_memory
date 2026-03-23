"""LLM client — supports HuggingFace local inference, vLLM, OpenAI API, Anthropic API.

Usage
-----
# Automatically picks backend from config.py:
from llm_client import create_llm_client
client = create_llm_client()

# Or construct a specific backend manually:
from llm_client import HFLocalClient, VLLMClient, OpenAIAPIClient
client = HFLocalClient("Qwen/Qwen3-7B")
client = VLLMClient("http://localhost:8000/v1")
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

from config import CONFIG


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseLLMClient(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    def generate(self,
                 prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking: Optional[bool] = None) -> str:
        """Generate a completion.

        Args:
            prompt:      Full prompt string.
            model:       Model name override (ignored for single-model backends).
            temperature: Sampling temperature.
            max_tokens:  Max new tokens to generate.
            thinking:    If not None, override the default thinking-mode setting
                         (Qwen3 only). True = enable <think> reasoning.

        Returns:
            Generated text string (thinking scratchpad stripped).
        """


# ---------------------------------------------------------------------------
# Backend 1: HuggingFace local — no server required
# ---------------------------------------------------------------------------

class HFLocalClient(BaseLLMClient):
    """Runs a HuggingFace model in-process.

    Loads the model once; subsequent calls reuse it.
    Supports Qwen3's thinking mode via the /think / /no_think suffix.

    Example
    -------
    client = HFLocalClient("Qwen/Qwen3-7B")
    response = client.generate("Solve: ...", thinking=False)
    """

    def __init__(self,
                 model_name: Optional[str] = None,
                 device_map: Optional[str] = None,
                 torch_dtype: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name or CONFIG["generation_model"]
        _dtype_str = torch_dtype or CONFIG.get("hf_torch_dtype", "bfloat16")
        _dtype = getattr(torch, _dtype_str)
        _device_map = device_map or CONFIG.get("hf_device_map", "auto")

        logging.info(f"Loading HF model {self.model_name} (dtype={_dtype_str}, device_map={_device_map})…")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=_dtype,
            device_map=_device_map,
        )
        self.model.eval()
        logging.info("Model loaded.")

        # Detect Qwen3 by model name so we can handle thinking mode
        self._is_qwen3 = "qwen3" in self.model_name.lower()

    def generate(self,
                 prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking: Optional[bool] = None) -> str:
        import torch

        # Qwen3 thinking-mode suffix
        if self._is_qwen3 and thinking is not None:
            prompt = prompt + (" /think" if thinking else " /no_think")

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_kwargs: dict = dict(
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)

        # Strip Qwen3 <think>…</think> scratchpad from the output
        if self._is_qwen3:
            response = _strip_thinking(response)

        return response.strip()


# ---------------------------------------------------------------------------
# Backend 2: vLLM (OpenAI-compatible server)
# ---------------------------------------------------------------------------

class VLLMClient(BaseLLMClient):
    """Calls a running vLLM server via the OpenAI-compatible /chat/completions endpoint.

    Start the server with:
        python -m vllm.entrypoints.openai.api_server \\
            --model Qwen/Qwen3-7B --port 8000 --enable-reasoning
    """

    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 default_model: Optional[str] = None):
        from openai import OpenAI

        self.base_url = base_url or CONFIG["vllm_base_url"]
        self.api_key  = api_key  or CONFIG["vllm_api_key"]
        self.default_model = default_model or CONFIG["generation_model"]
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._is_qwen3 = "qwen3" in self.default_model.lower()

    def generate(self,
                 prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking: Optional[bool] = None) -> str:
        model_name = model or self.default_model

        # Qwen3 thinking mode via suffix
        if self._is_qwen3 and thinking is not None:
            prompt = prompt + (" /think" if thinking else " /no_think")

        try:
            response = self._client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            if self._is_qwen3:
                text = _strip_thinking(text)
            return text.strip()
        except Exception as e:
            logging.error(f"vLLM generation failed: {e}")
            raise


# ---------------------------------------------------------------------------
# Backend 3: OpenAI API
# ---------------------------------------------------------------------------

class OpenAIAPIClient(BaseLLMClient):
    """Calls the OpenAI API (gpt-4o, gpt-4-turbo, etc.)."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 default_model: Optional[str] = None):
        from openai import OpenAI

        key = api_key or CONFIG.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY env var or config openai_api_key.")
        self.default_model = default_model or CONFIG.get("openai_model", "gpt-4o")
        self._client = OpenAI(api_key=key)

    def generate(self,
                 prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking: Optional[bool] = None) -> str:
        try:
            response = self._client.chat.completions.create(
                model=model or self.default_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return (response.choices[0].message.content or "").strip()
        except Exception as e:
            logging.error(f"OpenAI API generation failed: {e}")
            raise


# ---------------------------------------------------------------------------
# Backend 4: Anthropic API
# ---------------------------------------------------------------------------

class AnthropicAPIClient(BaseLLMClient):
    """Calls the Anthropic API (claude-opus-4-6, claude-sonnet-4-6, etc.)."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 default_model: Optional[str] = None):
        import anthropic

        key = api_key or CONFIG.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("Anthropic API key not set. Set ANTHROPIC_API_KEY env var or config anthropic_api_key.")
        self.default_model = default_model or CONFIG.get("anthropic_model", "claude-opus-4-6")
        self._client = anthropic.Anthropic(api_key=key)

    def generate(self,
                 prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking: Optional[bool] = None) -> str:
        try:
            message = self._client.messages.create(
                model=model or self.default_model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text.strip()
        except Exception as e:
            logging.error(f"Anthropic API generation failed: {e}")
            raise


# ---------------------------------------------------------------------------
# Role-aware wrapper — applies per-role thinking defaults
# ---------------------------------------------------------------------------

class RoleAwareLLMClient(BaseLLMClient):
    """Wraps any backend and applies role-specific defaults (e.g. thinking mode).

    This is the client the rest of the codebase should use.
    It routes each call through the inner client, setting thinking=True for
    alignment prompts and thinking=False for code-generation prompts based
    on config.

    The role is conveyed through the `model` argument using special suffixes:
        model="<name>:align"    → thinking=True  (if qwen3_thinking_for_alignment)
        model="<name>:generate" → thinking=False (if not qwen3_thinking_for_generation)
        model=None              → use config defaults (thinking disabled)

    Alternatively callers can pass `thinking=` explicitly, which takes precedence.
    """

    def __init__(self, inner: BaseLLMClient):
        self._inner = inner

    def generate(self,
                 prompt: str,
                 model: Optional[str] = None,
                 temperature: float = 0.0,
                 max_tokens: int = 4096,
                 thinking: Optional[bool] = None) -> str:
        # Parse role suffix from model string
        role = None
        actual_model = model
        if model and ":" in model:
            actual_model, role = model.rsplit(":", 1)

        # Determine thinking if not explicitly passed
        if thinking is None:
            if role == "align":
                thinking = CONFIG.get("qwen3_thinking_for_alignment", True)
            elif role == "generate":
                thinking = CONFIG.get("qwen3_thinking_for_generation", False)
            else:
                thinking = False  # safe default

        return self._inner.generate(
            prompt=prompt,
            model=actual_model if actual_model else None,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(backend: Optional[str] = None) -> RoleAwareLLMClient:
    """Create the appropriate LLM client based on config.

    Args:
        backend: Override config BACKEND. One of:
                 "hf_local", "vllm", "openai", "anthropic"

    Returns:
        RoleAwareLLMClient wrapping the selected backend.

    Example
    -------
    # Uses BACKEND from config.py (default: "hf_local")
    client = create_llm_client()

    # Force vLLM
    client = create_llm_client("vllm")

    # Force OpenAI API
    client = create_llm_client("openai")
    """
    backend = backend or CONFIG.get("backend", "hf_local")

    if backend == "hf_local":
        inner: BaseLLMClient = HFLocalClient()
    elif backend == "vllm":
        inner = VLLMClient()
    elif backend == "openai":
        inner = OpenAIAPIClient()
    elif backend == "anthropic":
        inner = AnthropicAPIClient()
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            "Choose from: hf_local, vllm, openai, anthropic"
        )

    logging.info(f"Created LLM client: backend={backend}")
    return RoleAwareLLMClient(inner)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>…</think> scratchpad from generated text."""
    import re
    # Strip the thinking block — keep only the content after </think>
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return cleaned.strip()
