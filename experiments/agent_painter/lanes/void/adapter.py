"""Reviewed VOID byte contract and native Qwen3 CPU adapter.

This module deliberately imports neither Torch nor Transformers until load_base
or structural model validation. Downloaded Python is retained as inert evidence.
The upstream byte mapping is preserved; malformed Unicode/UTF-8 is refused rather
than replaced. The versioned painter framing is new supervised task framing,
not an upstream chat template. See README.md and upstream/PROVENANCE.json.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
FRAMING_ID = "cybersand-void-task-v1"
TOKENIZER_CONTRACT = "cybersand-void-utf8-strict-v1"
MODEL_ID = "dotlabs/void.1"
REVISION = "9c2be5087772867d45961abd11f6b35110617563"
SOURCE_TOKENIZER_SHA256 = "279b9990f77b1eaedc7c1184fe13d60273a713ead856200f1fb9ae323c7b1e03"
SOURCE_TOKENIZER_CONFIG_SHA256 = "0ecbd4fd39e35125800d806af981ba0cacd8a7ba5f02c9e729c4ecc54af6d300"
MAX_SEQUENCE_TOKENS = 2048
SPECIAL_TOKENS = {256: "<pad>", 257: "<bos>", 258: "<eos>"}
TARGET_SHAPES = {
    "self_attn.q_proj": (640, 640),
    "self_attn.k_proj": (128, 640),
    "self_attn.v_proj": (128, 640),
    "self_attn.o_proj": (640, 640),
    "mlp.gate_proj": (1440, 640),
    "mlp.up_proj": (1440, 640),
    "mlp.down_proj": (640, 1440),
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _read_json(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key: " + key)
            value[key] = item
        return value

    def invalid(value):
        raise ValueError("non-finite JSON constant: " + value)

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CyberSandVoidByteTokenizer:
    """Audited equivalent of the pinned tokenizer's valid UTF-8 byte contract.

    Literal '<eos>', '<bos>' and '<pad>' are ordinary bytes. Only the integer
    IDs 258, 257 and 256 are specials. Ordinary encoding adds no BOS or EOS,
    matching upstream build_inputs_with_special_tokens. Invalid bytes are never
    silently turned into U+FFFD, and unpaired Unicode surrogates are refused.
    """

    vocab_size = 259
    model_max_length = MAX_SEQUENCE_TOKENS
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "left"
    clean_up_tokenization_spaces = False
    split_special_tokens = True
    is_fast = False
    chat_template = None
    pad_token, bos_token, eos_token = "<pad>", "<bos>", "<eos>"
    pad_token_id, bos_token_id, eos_token_id = 256, 257, 258
    unk_token = None
    unk_token_id = None
    all_special_ids = [256, 257, 258]

    @property
    def special_tokens_map(self):
        return {"bos_token": self.bos_token, "eos_token": self.eos_token, "pad_token": self.pad_token}

    def __len__(self):
        return self.vocab_size

    def get_vocab(self):
        return {**{f"<0x{value:02X}>": value for value in range(256)},
                **{token: value for value, token in SPECIAL_TOKENS.items()}}

    def encode(self, text, add_special_tokens=False):
        if not isinstance(text, str) or type(add_special_tokens) is not bool:
            raise ValueError("VOID encoding requires text and an explicit boolean special-token policy")
        # No automatic BOS/EOS even when add_special_tokens=True: upstream's
        # implementation also returns the original token list unchanged.
        return list(text.encode("utf-8", errors="strict"))

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        if type(skip_special_tokens) is not bool or clean_up_tokenization_spaces is not False:
            raise ValueError("VOID decoding forbids token cleanup")
        parts, pending = [], bytearray()

        def flush():
            if pending:
                parts.append(bytes(pending).decode("utf-8", errors="strict"))
                pending.clear()

        for value in token_ids:
            if type(value) is not int or not 0 <= value < self.vocab_size:
                raise ValueError("VOID token ID outside the reviewed 259-symbol vocabulary")
            if value < 256:
                pending.append(value)
            else:
                flush()
                if not skip_special_tokens:
                    parts.append(SPECIAL_TOKENS[value])
        flush()
        return "".join(parts)

    def _checkpoint_config(self):
        return {
            "tokenizer_class": type(self).__name__,
            "cybersand_tokenizer_contract": TOKENIZER_CONTRACT,
            "source_model_id": MODEL_ID,
            "source_revision": REVISION,
            "source_tokenizer_sha256": SOURCE_TOKENIZER_SHA256,
            "vocab_size": self.vocab_size,
            "model_max_length": self.model_max_length,
            "padding_side": self.padding_side,
            "clean_up_tokenization_spaces": False,
            "split_special_tokens": True,
            "utf8_errors": "strict",
            "automatic_bos_eos": False,
            **self.special_tokens_map,
            "unk_token": None,
        }

    @classmethod
    def from_pretrained(cls, directory):
        """Load only the exact upstream contract or our exact saved contract."""
        directory = Path(directory)
        result = cls()
        config_file = directory / "tokenizer_config.json"
        config = _read_json(config_file)
        if "cybersand_tokenizer_contract" in config:
            if config != result._checkpoint_config():
                raise ValueError("VOID saved tokenizer contract differs")
        elif _sha256(config_file) != SOURCE_TOKENIZER_CONFIG_SHA256:
            raise ValueError("VOID upstream tokenizer config differs from reviewed pin")
        if _read_json(directory / "byte_vocab.json") != result.get_vocab():
            raise ValueError("VOID byte vocabulary differs from reviewed mapping")
        if _read_json(directory / "special_tokens_map.json") != result.special_tokens_map:
            raise ValueError("VOID special-token mapping differs")
        return result

    def save_pretrained(self, directory):
        """Save tokenizer contract with an immediate real-file readback check."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        values = {
            "tokenizer_config.json": self._checkpoint_config(),
            "byte_vocab.json": self.get_vocab(),
            "special_tokens_map.json": self.special_tokens_map,
        }
        for name, value in values.items():
            (directory / name).write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
        loaded = type(self).from_pretrained(directory)
        for text in ("", "{\"version\":2}\n", "é 日本語 🤖\x00 <eos><bos><pad>"):
            if (loaded.encode(text) != self.encode(text)
                    or loaded.decode(loaded.encode(text)) != text):
                raise ValueError("VOID tokenizer serialization readback differs")
        if loaded.decode(self.all_special_ids) != self.decode(self.all_special_ids):
            raise ValueError("VOID serialized special-token readback differs")
        return tuple(str(directory / name) for name in values)


def _selected_contract(selected):
    expected = _read_json(HERE / "profile.json")
    for key in ("model_id", "revision", "subfolder", "architecture", "model_type", "license",
                "max_sequence_tokens", "vocabulary_size", "lora_targets", "required_hashes", "framing"):
        if selected.get(key) != expected[key]:
            raise ValueError("VOID selected profile differs from reviewed lane: " + key)
    if selected.get("trust_remote_code") is not False:
        raise ValueError("VOID never executes downloaded remote code")
    return expected


def _check_config(config):
    fields = {"model_type": "qwen3", "vocab_size": 259, "hidden_size": 640, "intermediate_size": 1440,
              "num_hidden_layers": 24, "num_attention_heads": 10, "num_key_value_heads": 2,
              "head_dim": 64, "max_position_embeddings": 2048, "pad_token_id": 256,
              "bos_token_id": 257, "eos_token_id": 258, "tie_word_embeddings": True}
    for name, expected in fields.items():
        if getattr(config, name, None) != expected:
            raise ValueError("VOID model configuration differs: " + name)
    if getattr(config, "auto_map", None):
        raise ValueError("VOID model config must not execute custom model code")


def _expected_modules():
    return {f"model.layers.{layer}.{suffix}": shape
            for layer in range(24) for suffix, shape in TARGET_SHAPES.items()}


def load_base(directory, selected):
    """Load reviewed local bytes through the native Transformers Qwen3 class."""
    expected = _selected_contract(selected)
    directory = Path(directory)
    # The caller also validates its full snapshot receipt/inventory. These hashes
    # make this narrow adapter fail closed when invoked independently.
    for name, expected_sha in expected["required_hashes"].items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or _sha256(path) != expected_sha:
            raise ValueError("VOID required source bytes differ: " + name)
    tokenizer = CyberSandVoidByteTokenizer.from_pretrained(directory)
    import torch
    from transformers import Qwen3ForCausalLM

    base = Qwen3ForCausalLM.from_pretrained(directory, local_files_only=True, trust_remote_code=False,
                                         dtype=torch.float32, attn_implementation="eager")
    if type(base).__name__ != "Qwen3ForCausalLM":
        raise ValueError("VOID requires native Qwen3ForCausalLM")
    _check_config(base.config)
    base.to("cpu")
    base.config.use_cache = False
    modules = dict(base.named_modules())
    for name, shape in _expected_modules().items():
        module = modules.get(name)
        if not isinstance(module, torch.nn.Linear) or tuple(module.weight.shape) != shape:
            raise ValueError("VOID native LoRA target shape differs: " + name)
    for parameter in base.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("VOID requires materialized CPU float32 model parameters")
    return base, tokenizer


def encode_prompt(tokenizer, prompt, system):
    if type(tokenizer) is not CyberSandVoidByteTokenizer:
        raise ValueError("VOID requires its reviewed strict byte tokenizer")
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(system, str) or not system.strip():
        raise ValueError("VOID task framing requires a nonempty instruction and request")
    prefix = FRAMING_ID + "\n" + _canonical({"instruction": system, "request": prompt}) + "\n"
    ids = tokenizer.encode(prefix, add_special_tokens=False)
    if len(ids) >= MAX_SEQUENCE_TOKENS:
        raise ValueError("VOID complete prompt leaves no context for a completion; no truncation")
    return ids


def encode_row(tokenizer, row, system, max_length):
    if type(max_length) is not int or not 1 <= max_length <= MAX_SEQUENCE_TOKENS:
        raise ValueError("VOID context must be bounded by its reviewed 2048 byte-token limit")
    prefix = encode_prompt(tokenizer, row["prompt"], system)
    completion = row["completion"]
    if not isinstance(completion, str) or not completion:
        raise ValueError("VOID requires the complete nonempty admitted JSON target")
    target = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = prefix + target
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds {max_length}; no truncation")
    return {"id": row["id"], "input_ids": ids, "labels": [-100] * len(prefix) + target,
            "prefix_ids": prefix, "boundary": len(prefix), "target_tokens": len(target),
            "prompt_bytes": len(row["prompt"].encode("utf-8", errors="strict")),
            "completion_bytes": len(completion.encode("utf-8", errors="strict"))}


def _trainable_inventory(model, selected, *, allow_meta=False):
    """Inspect projections; meta is an explicit private structural-test mode."""
    _selected_contract(selected)
    import torch

    if set(getattr(model, "peft_config", {})) != {"default"}:
        raise ValueError("VOID requires exactly one default PEFT adapter")
    config = model.peft_config["default"]
    if (config.r != 8 or config.lora_alpha != 16 or config.lora_dropout != 0.05
            or config.bias != "none" or config.modules_to_save
            or set(config.target_modules) != set(selected["lora_targets"])
            or str(config.task_type) not in {"CAUSAL_LM", "TaskType.CAUSAL_LM"}
            or getattr(config, "use_dora", False) or getattr(config, "use_rslora", False)
            or getattr(config, "rank_pattern", None) or getattr(config, "alpha_pattern", None)):
        raise ValueError("VOID LoRA recipe differs from reviewed rank8/alpha16/dropout0.05 projection-only recipe")
    base = model.get_base_model()
    if type(base).__name__ != "Qwen3ForCausalLM":
        raise ValueError("VOID PEFT base is not the reviewed native Qwen3 model")
    _check_config(base.config)
    expected_modules = _expected_modules()
    actual = {name: module for name, module in base.named_modules() if hasattr(module, "lora_A")}
    if set(actual) != set(expected_modules):
        raise ValueError("VOID LoRA target set differs from the 168 reviewed projections")
    expected_parameters = set()
    for name, shape in expected_modules.items():
        module = actual[name]
        if (not isinstance(module.base_layer, torch.nn.Linear) or tuple(module.base_layer.weight.shape) != shape
                or set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}
                or tuple(module.lora_A["default"].weight.shape) != (8, shape[1])
                or tuple(module.lora_B["default"].weight.shape) != (shape[0], 8)):
            raise ValueError("VOID LoRA tensor shape/adapter differs: " + name)
        for matrix in ("A", "B"):
            expected_parameters.add(f"{name}.lora_{matrix}.default.weight")
    trainable = {name: value for name, value in base.named_parameters() if value.requires_grad}
    if set(trainable) != expected_parameters:
        raise ValueError("VOID trainable weights include missing projections or unauthorized base/embedding/head weights")
    allowed_devices = {"cpu", "meta"} if allow_meta else {"cpu"}
    if any(value.dtype != torch.float32 or value.device.type not in allowed_devices
           for value in base.parameters()):
        raise ValueError("VOID model parameters require materialized CPU float32; meta is only for explicit private inspection")
    return {"architecture": type(base).__name__, "target_modules": len(actual),
            "trainable_tensors": len(trainable), "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "frozen_parameters": sum(p.numel() for p in base.parameters() if not p.requires_grad),
            "meta_parameters": sum(p.numel() for p in base.parameters() if p.device.type == "meta"),
            "base_embedding_and_head_frozen": True, "rank": 8, "alpha": 16, "dropout": 0.05}


def validate_trainable(model, selected):
    """Production hook: exact projection LoRA and materialized CPU float32 only."""
    return _trainable_inventory(model, selected)


def set_training_mode(model):
    model.train()
