"""Shared helpers for loading Hugging Face components with sane defaults.

Some embedding repositories ship custom modeling code and require
``trust_remote_code=True``. In non-interactive settings (e.g., slurm jobs) the
interactive prompt from `transformers` raises an EOFError. These helpers retry
with ``trust_remote_code=True`` when the error message explicitly requests it,
unless fallback is disabled.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

T = TypeVar("T")


def _load_with_remote_code_retry(
    loader: Callable[..., T],
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    allow_remote_code_fallback: bool = True,
    supports_safetensors: bool = False,
    component_label: str,
    **kwargs: Any,
) -> T:
    """Load a HF component, optionally retrying with trust_remote_code=True."""
    try:
        return loader(model_name_or_path, trust_remote_code=trust_remote_code, **kwargs)
    except ValueError as exc:
        message = str(exc)
        needs_remote_code = "trust_remote_code=True" in message
        needs_torch_upgrade = "upgrade torch to at least v2.6" in message

        if needs_remote_code and allow_remote_code_fallback and not trust_remote_code:
            print(
                f"{component_label} for {model_name_or_path} requires custom code; "
                "retrying with trust_remote_code=True."
            )
            # Retry once with remote code enabled; if that still fails, let it propagate.
            return _load_with_remote_code_retry(
                loader,
                model_name_or_path,
                trust_remote_code=True,
                allow_remote_code_fallback=False,
                supports_safetensors=supports_safetensors,
                component_label=component_label,
                **kwargs,
            )

        if supports_safetensors and needs_torch_upgrade:
            # Transformers blocks torch.load on torch<2.6 for unsafe .bin checkpoints.
            # Prefer safetensors if available to avoid the torch upgrade requirement.
            try:
                return loader(
                    model_name_or_path,
                    trust_remote_code=trust_remote_code,
                    use_safetensors=True,
                    **kwargs,
                )
            except Exception as safetensor_exc:
                raise RuntimeError(
                    "Model weights require torch>=2.6 or safetensors. "
                    "Install safetensors and/or upgrade torch, or use a safetensors "
                    "checkpoint for this model."
                ) from safetensor_exc

        raise


def load_tokenizer(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    allow_remote_code_fallback: bool = True,
    **kwargs: Any,
):
    return _load_with_remote_code_retry(
        AutoTokenizer.from_pretrained,
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        allow_remote_code_fallback=allow_remote_code_fallback,
        component_label="Tokenizer",
        **kwargs,
    )


def load_model(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    allow_remote_code_fallback: bool = True,
    **kwargs: Any,
):
    return _load_with_remote_code_retry(
        AutoModel.from_pretrained,
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        allow_remote_code_fallback=allow_remote_code_fallback,
        supports_safetensors=True,
        component_label="Model",
        **kwargs,
    )


def load_causal_lm_model(
    model_name_or_path: str,
    *,
    trust_remote_code: bool = False,
    allow_remote_code_fallback: bool = True,
    **kwargs: Any,
):
    return _load_with_remote_code_retry(
        AutoModelForCausalLM.from_pretrained,
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        allow_remote_code_fallback=allow_remote_code_fallback,
        supports_safetensors=True,
        component_label="Causal LM",
        **kwargs,
    )
