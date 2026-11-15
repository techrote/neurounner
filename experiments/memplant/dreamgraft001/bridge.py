"""G0 model-native bridge. Real cache readback; no claim of direct KV transplant.

Transformers 4.57 DynamicCache is supported, along with an explicit legacy
readback for regression tests. No fake/zero source cache is substituted.
"""
from __future__ import annotations

from hashlib import sha256

import torch
from torch import nn
from torch.nn import functional as F


def read_cache(cache):
    """Return [(K,V)] with shape [batch,kv_heads,time,head_dim]. Fail closed."""
    if hasattr(cache, "layers"):
        pairs = [(layer.keys, layer.values) for layer in cache.layers]
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache, strict=True))
    elif isinstance(cache, (tuple, list)):
        pairs = [(layer[0], layer[1]) for layer in cache]
    else:
        raise TypeError(f"Unsupported cache type: {type(cache).__name__}")
    if not pairs:
        raise ValueError("donor emitted no cache layers")
    for idx, (key, value) in enumerate(pairs):
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"Cache layer {idx} has no real tensor K/V")
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError(f"Cache layer {idx} has invalid geometry")
        if key.shape[0] != 1 or min(key.shape) == 0:
            raise ValueError(f"Cache layer {idx} invalid/empty")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError(f"Cache layer {idx} nonfinite")
    return pairs


def source_fingerprint(cache, *, bins=4, layers=4):
    """Return a fixed-size genuine V fingerprint + checked source geometry.

    Temporal binning is intentionally lossy. This is a contextual *feature*
    transfer experiment, not proof of addressable KV memory.
    """
    pairs = read_cache(cache)
    if len(pairs) < layers:
        raise ValueError("donor lacks required layers")
    selected = [round(i * (len(pairs) - 1) / (layers - 1)) for i in range(layers)]
    if len(set(selected)) != layers:
        raise ValueError("not enough distinct layers")
    features = []
    for idx in selected:
        value = pairs[idx][1].detach().float().cpu()
        time = value.shape[-2]
        if time < bins:
            raise ValueError("donor source shorter than temporal bins")
        pooled = []
        for j in range(bins):
            lo = (j * time) // bins
            hi = ((j + 1) * time) // bins
            if hi <= lo:
                raise ValueError("empty source time bin")
            pooled.append(value[:, :, lo:hi].mean(dim=2).reshape(-1))
        features.extend(pooled)
    fingerprint = torch.cat(features).contiguous()
    if not torch.isfinite(fingerprint).all():
        raise ValueError("nonfinite source fingerprint")
    geometry = {"cache_layers": len(pairs), "selected_layers": selected,
                "kv_heads": int(pairs[0][0].shape[1]),
                "tokens": int(pairs[0][0].shape[2]),
                "head_dim": int(pairs[0][0].shape[3]),
                "bins": bins, "feature_dim": int(fingerprint.numel()),
                "real_K_read": True, "real_V_read": True,
                "keys_nonzero": bool(torch.count_nonzero(pairs[0][0]) > 0),
                "values_nonzero": bool(torch.count_nonzero(pairs[0][1]) > 0)}
    if not geometry["keys_nonzero"] or not geometry["values_nonzero"]:
        raise ValueError("donor cache unexpectedly all zero")
    geometry["feature_sha256"] = sha256(fingerprint.numpy().tobytes()).hexdigest()
    return fingerprint, geometry


class DreamPrefix(nn.Module):
    """Small trainable feature → recipient soft tokens, frozen B parameters."""
    def __init__(self, source_dim: int, target_dim: int, *, slots=8, bottleneck=64, scale=0.05):
        super().__init__()
        if min(source_dim, target_dim, slots, bottleneck) <= 0:
            raise ValueError("invalid adapter geometry")
        self.slots, self.target_dim = slots, target_dim
        self.net = nn.Sequential(nn.LayerNorm(source_dim),
                                 nn.Linear(source_dim, bottleneck), nn.GELU(),
                                 nn.Linear(bottleneck, slots * target_dim))
        self.scale = float(scale)
        nn.init.normal_(self.net[-1].weight, std=0.002)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        if features.ndim != 2:
            raise ValueError("features must be [batch,dim]")
        return (self.scale * self.net(features)).view(features.shape[0], self.slots, self.target_dim)


def format_prompt(tokenizer, question):
    # Every transplant arm uses this *identical* question, not the donor source.
    enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=True,
        add_generation_prompt=True, return_tensors="pt")
    if isinstance(enc, dict):
        enc = enc["input_ids"]
    if enc.ndim != 2 or enc.shape[0] != 1 or enc.shape[1] == 0:
        raise RuntimeError("chat template produced invalid prompt")
    return enc


def answer_loss(receiver, prompt_ids, target_ids, prefix=None):
    """Gradient-preserving teacher-forced CE; receiver weights may be frozen.

    Prefix slots and question are masked, only answer tokens are supervised.
    Output logits for target position i come from input position i-1.
    """
    if target_ids.ndim != 2 or target_ids.shape[0] != 1 or not target_ids.numel():
        raise ValueError("invalid target ids")
    embedded = receiver.get_input_embeddings()
    prompt_emb = embedded(prompt_ids)
    answer_emb = embedded(target_ids)
    if prefix is None:
        parts = [prompt_emb, answer_emb]
        slots = 0
    else:
        if prefix.ndim != 3 or prefix.shape[0] != 1 or prefix.shape[2] != prompt_emb.shape[-1]:
            raise ValueError("invalid prefix geometry")
        slots = prefix.shape[1]
        parts = [prefix, prompt_emb, answer_emb]
    inputs = torch.cat(parts, dim=1)
    labels = torch.full((1, inputs.shape[1]), -100, dtype=torch.long, device=inputs.device)
    labels[:, slots + prompt_ids.shape[1]:] = target_ids
    mask = torch.ones((1, inputs.shape[1]), dtype=torch.long, device=inputs.device)
    logits = receiver(inputs_embeds=inputs, attention_mask=mask,
                      use_cache=False, return_dict=True).logits
    return F.cross_entropy(logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
                           labels[:, 1:].contiguous().view(-1), ignore_index=-100)


def generation_from_prefix(receiver, tokenizer, prompt_ids, prefix, *, max_tokens=16):
    """G0 independent inference-cache continuation check, not creative scoring.

    Manual decoding avoids the HF generate(inputs_embeds=...) API-version traps.
    Explicit first-token prefill and one-token recurrent forwards verify cache
    persistence/positions. This is a freshly constructed B-native cache.
    """
    with torch.no_grad():
        embeds = receiver.get_input_embeddings()(prompt_ids)
        initial = torch.cat([prefix, embeds], dim=1)
        mask = torch.ones((1, initial.shape[1]), dtype=torch.long)
        out = receiver(inputs_embeds=initial, attention_mask=mask,
                       use_cache=True, return_dict=True)
        cache = out.past_key_values
        predicted = []
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        for i in range(max_tokens):
            predicted.append(nxt)
            if int(nxt.item()) == tokenizer.eos_token_id:
                break
            mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
            # Cache positions advance across the *total* prefix+query.
            out = receiver(input_ids=nxt, attention_mask=mask,
                           past_key_values=cache, use_cache=True, return_dict=True)
            cache = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat(predicted, dim=1)
        return tokenizer.decode(ids[0].tolist(), skip_special_tokens=True), ids[0].tolist()
