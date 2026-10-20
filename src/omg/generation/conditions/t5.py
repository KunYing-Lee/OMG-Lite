from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn

from omg.core.paths import resolve_repo_path


class FrozenT5TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "t5-base",
        max_length: int = 50,
        output_dim: int = 768,
        frozen_eval: bool = False,
        cache_max_entries: int = 0,
        cache_device: str = "cpu",
    ):
        super().__init__()
        self.frozen_eval = bool(frozen_eval)
        self.cache_max_entries = int(cache_max_entries)
        if cache_device not in {"cpu", "encoder"}:
            raise ValueError("cache_device must be cpu or encoder")
        self.cache_device = cache_device
        if self.cache_max_entries < 0:
            raise ValueError("cache_max_entries must be nonnegative")
        if self.cache_max_entries and not self.frozen_eval:
            raise ValueError("Text caching requires frozen_eval=True; stochastic dropout cannot be memoized")
        self._text_cache = OrderedDict()
        self._cache_signature = None
        self.cache_hits = 0
        self.cache_misses = 0
        try:
            from transformers import T5EncoderModel, T5Tokenizer
        except ImportError as exc:
            raise ImportError(
                "FrozenT5TextEncoder requires the `transformers` package. "
                "Install OMG with the training extras."
            ) from exc

        self.model_name = self._resolve_model_name(str(model_name))
        self.max_length = int(max_length)
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self.encoder = T5EncoderModel.from_pretrained(self.model_name)
        except OSError as exc:
            raise OSError(
                f"Failed to load T5 text encoder from {self.model_name!r}. "
                "Download t5-base to models/t5-base-local, set OMG_MODELS_ROOT, "
                "or override model.text_encoder.model_name with a valid local path or Hugging Face model id."
            ) from exc
        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad_(False)

        hidden_dim = int(self.encoder.config.d_model)
        self.output_dim = int(output_dim)
        if hidden_dim == self.output_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(hidden_dim, self.output_dim)

    @staticmethod
    def _resolve_model_name(model_name: str) -> str:
        path = Path(model_name).expanduser()
        first_part = path.parts[0] if path.parts else ""
        is_local_path = (
            path.is_absolute()
            or str(path).startswith((".", "~"))
            or first_part in {"models", "assets", "outputs", "checkpoints"}
            or path.exists()
        )
        if not is_local_path:
            return model_name
        resolved = path if path.is_absolute() else resolve_repo_path(path)
        if not resolved.exists():
            raise FileNotFoundError(
                f"T5 text encoder path does not exist: {resolved}. "
                "Download t5-base to models/t5-base-local, set OMG_MODELS_ROOT, "
                "or override model.text_encoder.model_name with a valid local path or Hugging Face model id."
            )
        return str(resolved)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_eval:
            self.encoder.eval()
        return self

    def clear_cache(self) -> None:
        self._text_cache.clear()
        self._cache_signature = None
        self.cache_hits = 0
        self.cache_misses = 0

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _encode(self, captions: list[str], device: torch.device):
        tokenized = self.tokenizer(
            captions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        with torch.no_grad():
            hidden = self.encoder(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
            ).last_hidden_state
        return hidden, tokenized["attention_mask"].bool()

    def _cached_encode(self, captions: list[str], device: torch.device):
        if self.encoder.training:
            raise RuntimeError("Cannot cache T5 outputs with encoder in training mode")
        parameter = next(self.encoder.parameters())
        autocast = torch.is_autocast_enabled(device.type)
        signature = (device, parameter.dtype, autocast,
                     torch.get_autocast_dtype(device.type) if autocast else None,
                     self.max_length, self.cache_device)
        if signature != self._cache_signature:
            self.clear_cache()
            self._cache_signature = signature
        # Per-call entries survive LRU eviction, including batches larger than capacity.
        entries = {}
        missing = []
        for caption in dict.fromkeys(captions):
            if caption in self._text_cache:
                entries[caption] = self._text_cache[caption]
                self._text_cache.move_to_end(caption)
            else:
                missing.append(caption)
        self.cache_misses += len(missing)
        self.cache_hits += len(captions) - len(missing)
        if missing:
            hidden, mask = self._encode(missing, device)
            hidden = hidden.detach()
            if self.cache_device == "cpu":
                hidden, mask = hidden.cpu(), mask.cpu()
            for index, caption in enumerate(missing):
                # Clone each row so eviction really releases its batch storage.
                entry = (hidden[index].clone(), mask[index].clone())
                entries[caption] = entry
                self._text_cache[caption] = entry
                while len(self._text_cache) > self.cache_max_entries:
                    self._text_cache.popitem(last=False)
        hidden = torch.stack([entries[text][0] for text in captions]).to(device)
        mask = torch.stack([entries[text][1] for text in captions]).to(device)
        return hidden, mask

    def forward(
        self,
        captions: list[str],
        has_text: torch.Tensor | None = None,
        force_null_text: bool = False,
        device: torch.device | None = None,
    ) -> dict[str, torch.Tensor]:
        if force_null_text:
            captions = ["" for _ in captions]
        if len(captions) == 0:
            raise ValueError("FrozenT5TextEncoder received an empty caption batch")

        if device is None:
            device = next(self.parameters()).device
        device = torch.device(device)
        if self.cache_max_entries:
            hidden, mask = self._cached_encode(captions, device)
        else:
            hidden, mask = self._encode(captions, device)
        # The optional trainable projection and random CFG mask stay outside cache.
        hidden = self.proj(hidden)
        if has_text is not None:
            has_text = has_text.to(device=device, dtype=torch.bool).view(-1, 1)
            mask = mask & has_text
            hidden = hidden * mask.unsqueeze(-1).to(hidden.dtype)
        return {"context": hidden, "mask": mask}
