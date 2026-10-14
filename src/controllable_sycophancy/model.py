from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class GenerationBackend(ABC):
    """
    Common interface for any inference backend.

    The rest of the research code should not care whether generation happens
    through:

    - local Hugging Face / PyTorch,
    - MLX via an OpenAI-compatible local server,
    - the official OpenAI API.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        raise NotImplementedError


def load_tokenizer(
    model_name: str,
    *,
    trust_remote_code: bool = False,
):
    """
    Retained for compatibility with training and legacy local inference code.
    """

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_causal_lm(
    model_name: str,
    *,
    trust_remote_code: bool = False,
    quantized: bool = False,
):
    """
    Load a Hugging Face causal language model.

    On CUDA, quantized=True attempts 4-bit loading.

    On non-CUDA machines we deliberately avoid device_map='auto',
    because it can behave poorly on macOS.
    """

    if torch.cuda.is_available():
        kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "device_map": "auto",
        }

        if quantized:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except Exception:
                kwargs["dtype"] = torch.float16
        else:
            kwargs["dtype"] = torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **kwargs,
        )

        return model

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        dtype=torch.float32,
    )

    model.to("cpu")

    return model


def apply_chat_template(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool = True,
) -> str:
    """
    Apply a tokenizer's chat template.

    Qwen3 supports enable_thinking=False, but not every tokenizer exposes
    this argument, so we fall back gracefully.
    """

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )


class TransformersBackend(GenerationBackend):
    """
    Hugging Face / PyTorch inference backend.

    Kept so the repository still works on CUDA machines.
    """

    def __init__(
        self,
        *,
        model_name: str,
        trust_remote_code: bool = False,
        quantized: bool = True,
    ) -> None:
        self._model_name = model_name

        self.tokenizer = load_tokenizer(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        self.model = load_causal_lm(
            model_name,
            trust_remote_code=trust_remote_code,
            quantized=quantized,
        )

        self.model.eval()

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        rendered = apply_chat_template(
            self.tokenizer,
            messages,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
        ).to(self.model.device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                **generation_kwargs,
            )

        generated_tokens = output[
            0,
            inputs["input_ids"].shape[-1] :
        ]

        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()


class OpenAICompatibleBackend(GenerationBackend):
    """
    Backend for OpenAI-compatible chat-completions servers.

    This is what we use for mlx_lm.server on Apple Silicon.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_base: str,
        api_key: str = "local",
        disable_thinking: bool = True,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for the "
                "openai_compatible backend. Run:\n"
                "python -m pip install -U openai"
            ) from exc

        self._model_name = model_name
        self.disable_thinking = disable_thinking

        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_new_tokens,
        }

        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p
        else:
            # Some Qwen3 deployments behave poorly under true greedy
            # decoding. A tiny temperature gives deterministic-enough
            # judge behavior without invoking pathological greedy output.
            kwargs["temperature"] = 0.01
            kwargs["top_p"] = 1.0

        if self.disable_thinking:
            kwargs["extra_body"] = {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                }
            }

        try:
            response = self.client.chat.completions.create(
                **kwargs
            )

        except Exception as exc:
            raise RuntimeError(
                "Could not reach the OpenAI-compatible model server. "
                "Check that mlx_lm.server is running and that "
                "configs/model.example.yaml contains the correct api_base."
            ) from exc

        content = response.choices[0].message.content

        if content is None:
            return ""

        text = content.strip()

        # Defensive cleanup in case a Qwen3 endpoint emits thinking tags
        # despite enable_thinking=False.
        if "<think>" in text and "</think>" in text:
            _, _, after = text.partition("</think>")
            text = after.strip()

        return text


class OpenAIBackend(GenerationBackend):
    """
    Backend for the official OpenAI API.

    Intended for cloud-based evaluation and other tasks where we want
    a stronger independent model than the local participant generator.
    """

    def __init__(
        self,
        *,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is required for the "
                "OpenAI backend. Run:\n"
                "python -m pip install -U openai"
            ) from exc

        api_key = os.environ.get(api_key_env)

        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set."
            )

        self._model_name = model_name
        self.client = OpenAI(api_key=api_key)

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        *,
        messages: list[dict[str, str]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": messages,
            "max_completion_tokens": max_new_tokens,
        }

        if do_sample:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = top_p

        try:
            response = self.client.chat.completions.create(
                **kwargs
            )

        except Exception as exc:
            raise RuntimeError(
                "Could not reach the OpenAI API. "
                "Check that OPENAI_API_KEY is set and "
                "that API billing is enabled."
            ) from exc

        content = response.choices[0].message.content

        if content is None:
            return ""

        return content.strip()


def build_generation_backend(
    model_config: dict[str, Any],
) -> GenerationBackend:
    """
    Construct the configured inference backend.
    """

    backend_name = str(
        model_config.get(
            "backend",
            "transformers",
        )
    ).lower()

    model_name = str(
        model_config["base_model"]
    )

    if backend_name == "transformers":
        return TransformersBackend(
            model_name=model_name,
            trust_remote_code=bool(
                model_config.get(
                    "trust_remote_code",
                    False,
                )
            ),
            quantized=bool(
                model_config.get(
                    "quantized",
                    True,
                )
            ),
        )

    if backend_name == "openai_compatible":
        return OpenAICompatibleBackend(
            model_name=model_name,
            api_base=str(
                model_config.get(
                    "api_base",
                    "http://127.0.0.1:8080/v1",
                )
            ),
            api_key=str(
                model_config.get(
                    "api_key",
                    "local",
                )
            ),
            disable_thinking=bool(
                model_config.get(
                    "disable_thinking",
                    True,
                )
            ),
        )

    if backend_name == "openai":
        return OpenAIBackend(
            model_name=model_name,
            api_key_env=str(
                model_config.get(
                    "api_key_env",
                    "OPENAI_API_KEY",
                )
            ),
        )

    raise ValueError(
        f"Unknown inference backend: {backend_name}"
    )
