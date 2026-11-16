"""Pinned root Sprocket Instruct adapter; importing this file never imports Torch.

The shared trainer owns snapshot verification, numerical setup, optimization,
checkpoint custody and native admission. This lane owns model/framing guards.
"""
from hashlib import sha256
from pathlib import Path

from common import encode_row as common_encode_row, read_json


PROFILE = read_json(Path(__file__).with_name("profile.json"))
SPECIAL_TOKENS = (
    "<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>",
    "<|assistant|>", "<|end|>", "<|tool_call|>", "<|tool_result|>",
    "<think>", "</think>", "<|memory_read|>", "<|memory_write|>",
    "<|reserved_4|>", "<|reserved_5|>", "<|reserved_6|>", "<|reserved_7|>",
)


def validate_profile(selected):
    """A generic Llama or the sibling chat checkpoint cannot use this adapter."""
    for key in ("model_id", "revision", "subfolder", "loader", "model_type",
                "architecture", "trust_remote_code", "license", "lora_targets",
                "max_sequence_tokens", "required_hashes", "model_config", "framing"):
        if selected.get(key) != PROFILE[key]:
            raise ValueError(f"Sprocket root Instruct profile mismatch: {key}")


def validate_tokenizer(tokenizer):
    template = tokenizer.chat_template
    if not isinstance(template, str) or sha256(template.encode()).hexdigest() != PROFILE["framing"]["template_sha256"]:
        raise ValueError("Sprocket requires the exact upstream root Jinja template")
    if (len(tokenizer), tokenizer.bos_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) != (32000, 0, 1, 5):
        raise ValueError("Sprocket vocabulary/BOS/PAD/EOS identity mismatch")
    for index, token in enumerate(SPECIAL_TOKENS):
        if tokenizer.convert_tokens_to_ids(token) != index or tokenizer.encode(token, add_special_tokens=False) != [index]:
            raise ValueError(f"Sprocket special-token identity mismatch: {token}")


def _content_ids(tokenizer, content):
    if not isinstance(content, str) or not content:
        raise ValueError("Sprocket message content must be a nonempty string")
    ids = tokenizer.encode(content, add_special_tokens=False)
    if not ids or any(token < len(SPECIAL_TOKENS) for token in ids):
        raise ValueError("Sprocket message content contains a control or unknown token")
    return ids


def encode_prompt(tokenizer, prompt, system):
    """Use the upstream template, with no synthetic BOS or chat-build substitution."""
    validate_tokenizer(tokenizer)
    system_ids = _content_ids(tokenizer, system)
    prompt_ids = _content_ids(tokenizer, prompt)
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    expected = [2] + system_ids + [5, 3] + prompt_ids + [5, 4]
    if ids != expected:
        raise ValueError("Sprocket upstream prompt framing drift")
    if len(ids) >= PROFILE["max_sequence_tokens"]:
        raise ValueError("Sprocket prompt leaves no context for a completion; no truncation")
    return ids


def encode_row(tokenizer, row, system, max_length):
    """Complete JSON and one true EOS are supervised; every prompt token is masked."""
    if type(max_length) is not int or not 1 <= max_length <= PROFILE["max_sequence_tokens"]:
        raise ValueError("Sprocket sequence limit must be in 1..2048")
    prefix = encode_prompt(tokenizer, row["prompt"], system)
    completion = _content_ids(tokenizer, row["completion"])
    encoded = common_encode_row(tokenizer, row, system, max_length)
    if encoded["prefix_ids"] != prefix or encoded["boundary"] != len(prefix):
        raise ValueError("Sprocket training and inference prefixes differ")
    if encoded["input_ids"] != prefix + completion + [5]:
        raise ValueError("Sprocket completion framing or terminal EOS changed")
    if encoded["labels"] != [-100] * len(prefix) + completion + [5]:
        raise ValueError("Sprocket completion-only loss mask changed")
    if tokenizer.decode(completion, skip_special_tokens=False) != row["completion"]:
        raise ValueError("Sprocket complete target bytes do not round-trip")
    return encoded


def _validate_base(model):
    if type(model).__name__ != PROFILE["architecture"] or model.config.model_type != PROFILE["model_type"]:
        raise ValueError("Sprocket model class differs from the pinned root checkpoint")
    for key, value in PROFILE["model_config"].items():
        if getattr(model.config, key, None) != value:
            raise ValueError(f"Sprocket base architecture mismatch: {key}")
    if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
        raise ValueError("Sprocket requires tied input/output embeddings")


def _projection_shapes():
    shapes = {"q_proj": (1280, 1280), "k_proj": (256, 1280),
              "v_proj": (256, 1280), "o_proj": (1280, 1280),
              "gate_proj": (3584, 1280), "up_proj": (3584, 1280),
              "down_proj": (1280, 3584)}
    result = {}
    for layer in range(26):
        for target, shape in shapes.items():
            block = "self_attn" if target in ("q_proj", "k_proj", "v_proj", "o_proj") else "mlp"
            result[f"model.layers.{layer}.{block}.{target}"] = shape
    return result


def load_base(directory, selected):
    # The caller must establish the shared portable CPU recipe before this import.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train import verify_model

    validate_profile(selected)
    verify_model(directory, selected)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    validate_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
        dtype=torch.float32, attn_implementation="eager")
    model.to("cpu")
    model.config.use_cache = False
    _validate_base(model)
    if sum(parameter.numel() for parameter in model.parameters()) != PROFILE["inventory"]["base_parameters"]:
        raise ValueError("Sprocket base parameter inventory differs")
    for name, shape in _projection_shapes().items():
        module = model.get_submodule(name)
        if type(module) is not torch.nn.Linear or tuple(module.weight.shape) != shape or module.bias is not None:
            raise ValueError(f"Sprocket projection inventory differs: {name}")
    if any(parameter.dtype != torch.float32 or parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Sprocket base must be CPU FP32")
    return model, tokenizer


def _trainable_inventory(model, selected):
    """Shape-only inspection also supports meta-device preflight, never training."""
    import torch
    from peft.tuners.lora.layer import Linear as LoraLinear

    validate_profile(selected)
    if set(model.peft_config) != {"default"}:
        raise ValueError("Sprocket requires exactly one default PEFT configuration")
    config = model.peft_config["default"]
    for key, value in {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                       "bias": "none", "task_type": "CAUSAL_LM", "modules_to_save": None,
                       "use_dora": False, "use_rslora": False, "rank_pattern": {},
                       "alpha_pattern": {}, "lora_bias": False}.items():
        if getattr(config, key, None) != value:
            raise ValueError(f"Sprocket exact PEFT recipe differs: {key}")
    if set(config.target_modules) != set(PROFILE["lora_targets"]):
        raise ValueError("Sprocket exact PEFT target_modules differ")
    base = model.get_base_model()
    _validate_base(base)
    expected = _projection_shapes()
    adapted = {name: module for name, module in base.named_modules() if isinstance(module, LoraLinear)}
    if set(adapted) != set(expected):
        raise ValueError("Sprocket LoRA module whitelist differs from all 182 language projections")
    parameter_specs = {}
    ranks = set()
    for name, module in adapted.items():
        output_size, input_size = expected[name]
        if type(module.base_layer) is not torch.nn.Linear or tuple(module.base_layer.weight.shape) != expected[name]:
            raise ValueError(f"Sprocket LoRA base projection differs: {name}")
        if set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"} or module.use_dora.get("default", False):
            raise ValueError("Sprocket requires exactly one ordinary default LoRA adapter")
        rank = module.r["default"]
        if type(rank) is not int or rank != 8:
            raise ValueError("Sprocket LoRA rank must be exactly 8")
        if module.lora_alpha["default"] != 16 or module.scaling["default"] != 2.0 or module.lora_dropout["default"].p != 0.05:
            raise ValueError("Sprocket actual LoRA alpha, scaling or dropout differs")
        ranks.add(rank)
        for matrix, shape in (("A", (rank, input_size)), ("B", (output_size, rank))):
            key = f"base_model.model.{name}.lora_{matrix}.default.weight"
            parameter_specs[key] = shape
    actual = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(ranks) != 1 or set(actual) != set(parameter_specs):
        raise ValueError("Sprocket trainable tensors must be only the complete uniform-rank LoRA whitelist")
    for name, parameter in actual.items():
        if tuple(parameter.shape) != parameter_specs[name] or parameter.dtype != torch.float32:
            raise ValueError(f"Sprocket trainable tensor shape/dtype differs: {name}")
    return {"architecture": type(base).__name__, "adapter": "default",
            "lora_targets": list(selected["lora_targets"]), "lora_modules": len(adapted),
            "lora_rank": next(iter(ranks)), "lora_alpha": 16, "lora_dropout": 0.05,
            "trainable_tensors": len(actual),
            "trainable_parameters": sum(parameter.numel() for parameter in actual.values()),
            "frozen_parameters": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
            "trainable_parameter_names": sorted(actual)}


def validate_trainable(model, selected):
    result = _trainable_inventory(model, selected)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Sprocket training requires materialized CPU parameters; meta inspection is not training")
    if result["frozen_parameters"] != PROFILE["inventory"]["base_parameters"]:
        raise ValueError("Sprocket full frozen base inventory differs")
    result["device"] = "cpu"
    result["dtype"] = "float32"
    return result


def set_training_mode(model):
    model.config.use_cache = False
    model.train()
