"""Pinned, text-only adaptation of the actual LFM2.5-VL multimodal checkpoint.

Importing this module does not import Torch. The common runner establishes its
CPU numerical profile before calling load_base. There is no downloaded Python,
model substitution, image input, or image/connector adapter in this lane.
"""

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROFILE = json.loads((HERE / "profile.json").read_text(encoding="utf-8"))
MODEL_ID = "LiquidAI/LFM2.5-VL-450M"
REVISION = "fc6221ca597f3315e4f82fc2df606783267b34ba"
FRAMING_VERSION = "lfm25-vl-text-only-end-at-eos-v1"
LANGUAGE_PREFIX = "model.language_model.layers."
FROZEN_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")


def _hash_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected(selected):
    for name in ("model_id", "revision", "subfolder", "loader", "model_type",
                 "architecture", "trust_remote_code", "license", "card_license",
                 "license_name", "license_link", "max_sequence_tokens",
                 "lora_targets", "lora_target_paths", "required_hashes", "framing", "expected_inventory"):
        if selected.get(name) != PROFILE[name]:
            raise ValueError("LFM immutable profile mismatch: " + name)
    return selected


def _check_environment():
    for name, expected in PROFILE["implementation_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("unqualified LFM package version: " + name)
    # These optional packages replace the reviewed Torch convolution fallback.
    # They are absent from the pinned CPU environment; do not select them silently.
    for package in ("causal_conv1d", "kernels"):
        if importlib.util.find_spec(package) is not None:
            raise ValueError("unqualified LFM optional kernel package: " + package)


def _check_files(directory, *, weights):
    directory = Path(directory)
    for name, expected in PROFILE["required_hashes"].items():
        if name == "model.safetensors" and not weights:
            continue
        path = directory / name
        if not path.is_file() or path.is_symlink() or _hash_file(path) != expected:
            raise ValueError("LFM snapshot hash mismatch: " + name)
    return directory


class TextOnlyProcessor:
    """Small text-facing interface; saves the real multimodal processor intact."""

    def __init__(self, processor, snapshot):
        if type(processor).__name__ != "Lfm2VlProcessor":
            raise ValueError("actual Lfm2VlProcessor required")
        self.processor = processor
        self.tokenizer = processor.tokenizer
        if type(self.tokenizer).__name__ != "TokenizersBackend":
            raise ValueError("unexpected official LFM tokenizer class")
        if type(processor.image_processor).__name__ != "Lfm2VlImageProcessor":
            raise ValueError("unexpected official LFM image processor class")
        expected = PROFILE["framing"]
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if getattr(self.tokenizer, name) != expected[name]:
                raise ValueError("LFM special token mismatch: " + name)
        if (len(self.tokenizer) != expected["tokenizer_size"]
                or self.tokenizer.convert_tokens_to_ids("<image>") != expected["image_token_id"]):
            raise ValueError("LFM tokenizer vocabulary/image identity mismatch")
        if (not isinstance(processor.chat_template, str)
                or hashlib.sha256(processor.chat_template.encode()).hexdigest()
                != expected["template_sha256"]):
            raise ValueError("official LFM chat template changed")
        self._upstream_license = (Path(snapshot) / "LICENSE").read_bytes()
        self._upstream_readme = (Path(snapshot) / "README.md").read_bytes()
        for name, data in (("LICENSE", self._upstream_license), ("README.md", self._upstream_readme)):
            if hashlib.sha256(data).hexdigest() != PROFILE["required_hashes"][name]:
                raise ValueError("upstream attribution hash mismatch: " + name)

    @property
    def chat_template(self):
        return self.processor.chat_template

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bos_token_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_token(self):
        return self.tokenizer.eos_token

    def __len__(self):
        return len(self.tokenizer)

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        if tokenize:
            raise ValueError("render LFM framing before encoding")
        if (not isinstance(messages, list) or not messages
                or any(set(message) != {"role", "content"}
                       or message["role"] not in {"system", "user", "assistant"}
                       or not isinstance(message["content"], str) for message in messages)):
            raise ValueError("LFM lane accepts only explicit text messages")
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt,
            tools=None, keep_past_thinking=False, continue_final_message=False,
        )
        if messages[-1]["role"] == "assistant" and not add_generation_prompt:
            if not rendered.endswith(self.eos_token + "\n"):
                raise ValueError("unexpected official LFM assistant suffix")
            # Remove the template separator after EOS, never answer bytes/tokens.
            rendered = rendered[:-1]
        return rendered

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("LFM template owns its BOS and role tokens")
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if any(token in token_ids for token in (396, 498, 499)):
            raise ValueError("image tokens are forbidden in the text-only LFM lane")
        encoded = self.processor(
            text=text, images=None, add_special_tokens=False,
            padding=False, truncation=False,
        )
        ids = encoded["input_ids"]
        if len(ids) != 1 or not isinstance(ids[0], list):
            raise ValueError("expected one unpadded LFM text sequence")
        ids = ids[0]
        if ids != token_ids:
            raise ValueError("LFM processor/tokenizer text framing differs")
        return ids

    def decode(self, ids, **kwargs):
        kwargs.setdefault("clean_up_tokenization_spaces", False)
        return self.tokenizer.decode(ids, **kwargs)

    def save_pretrained(self, directory, **kwargs):
        from transformers import Lfm2VlProcessor
        directory = Path(directory)
        self.processor.save_pretrained(directory, **kwargs)
        restored = Lfm2VlProcessor.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False,
        )
        if (restored.chat_template != self.chat_template
                or restored.tokenizer.special_tokens_map != self.tokenizer.special_tokens_map
                or json.loads(restored.tokenizer.backend_tokenizer.to_str())
                != json.loads(self.tokenizer.backend_tokenizer.to_str())
                or restored.image_processor.to_dict() != self.processor.image_processor.to_dict()):
            raise ValueError("LFM processor serialization/readback differs")
        # The upstream licence applies to model derivatives, including adapters.
        (directory / "LICENSE").write_bytes(self._upstream_license)
        (directory / "UPSTREAM_README.md").write_bytes(self._upstream_readme)
        notice = {
            "model_id": MODEL_ID, "revision": REVISION,
            "license": PROFILE["license"], "framing": FRAMING_VERSION,
            "modification": "CyberSand synthetic procedural-program language-side LoRA; frozen vision and connector.",
            "processor": "Official Lfm2VlProcessor serialized without changes to its chat template.",
            "processor_semantic_readback": "PASS: complete tokenizer backend, special tokens, template and image processor configuration",
        }
        (directory / "LFM_ADAPTER_NOTICE.json").write_text(
            json.dumps(notice, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )


def load_processor(directory, selected=None):
    """Small-file preflight and the same processor used by hosted training."""
    _selected(PROFILE if selected is None else selected)
    _check_environment()
    directory = _check_files(directory, weights=False)
    from transformers import Lfm2VlProcessor
    processor = Lfm2VlProcessor.from_pretrained(
        directory, local_files_only=True, trust_remote_code=False,
    )
    return TextOnlyProcessor(processor, directory)


def _base_model(model):
    base = model.get_base_model() if hasattr(model, "peft_config") else model
    if type(base).__name__ != PROFILE["architecture"]:
        raise ValueError("actual Lfm2VlForConditionalGeneration required")
    config = base.config
    if (config.model_type != "lfm2_vl" or config.text_config.model_type != "lfm2"
            or config.vision_config.model_type != "siglip2_vision_model"
            or config.text_config.vocab_size != 65536):
        raise ValueError("wrong multimodal LFM configuration")
    for component, expected in ((base.model.vision_tower, "Siglip2VisionModel"),
                                (base.model.multi_modal_projector, "Lfm2VlMultiModalProjector"),
                                (base.model.language_model, "Lfm2Model")):
        if type(component).__name__ != expected:
            raise ValueError("multimodal component substitution: " + expected)
    if any(getattr(component, "_use_kernels", False) for component in base.modules()):
        raise ValueError("unqualified LFM hub kernels")
    if (config._attn_implementation != "eager"
            or config.text_config._attn_implementation != "eager"
            or config.vision_config._attn_implementation != "eager"):
        raise ValueError("LFM CPU profile requires eager attention")
    return base


def _linear_inventory(base):
    import torch
    return {name: [module.in_features, module.out_features]
            for name, module in base.named_modules()
            if name.startswith(LANGUAGE_PREFIX) and isinstance(module, torch.nn.Linear)}


def load_base(directory, selected):
    """Load full official VL weights in FP32, never an extracted text checkpoint."""
    _selected(selected)
    _check_environment()
    directory = _check_files(directory, weights=True)
    import torch
    from transformers import Lfm2VlForConditionalGeneration
    processor = load_processor(directory, selected)
    model, loading_info = Lfm2VlForConditionalGeneration.from_pretrained(
        directory, dtype=torch.float32, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False, use_safetensors=True,
        output_loading_info=True,
    )
    if any(loading_info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError("LFM checkpoint did not load every expected parameter: " + repr(loading_info))
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    base = _base_model(model)
    inventory = _linear_inventory(base)
    if set(inventory) != set(PROFILE["lora_target_paths"]):
        raise ValueError("LFM language Linear module inventory changed")
    if sum(parameter.numel() for parameter in base.parameters()) != PROFILE["expected_inventory"]["base_parameters"]:
        raise ValueError("LFM full multimodal parameter inventory changed")
    for parameter in base.parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError("LFM lane requires CPU FP32 full base")
        parameter.requires_grad_(False)
    base.model.vision_tower.eval()
    base.model.multi_modal_projector.eval()
    return model, processor


def encode_prompt(tokenizer, prompt, system):
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    return tokenizer.encode(tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    ), add_special_tokens=False)


def encode_row(tokenizer, row, system, max_length):
    """Complete JSON target, exact training prefix, completion-only loss and EOS."""
    if type(max_length) is not int or not 1 <= max_length <= PROFILE["max_sequence_tokens"]:
        raise ValueError("LFM sequence limit must be within the qualified profile")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": row["prompt"]}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": row["completion"]}],
        tokenize=False, add_generation_prompt=False,
    )
    if not full.startswith(prefix) or not full.endswith(tokenizer.eos_token):
        raise ValueError("unstable LFM prompt/completion framing")
    ids = tokenizer.encode(full, add_special_tokens=False)
    prefix_ids = encode_prompt(tokenizer, row["prompt"], system)
    boundary = 0
    for left, right in zip(ids, prefix_ids):
        if left != right:
            break
        boundary += 1
    if len(prefix_ids) - boundary > 2:
        raise ValueError("unstable LFM BPE prompt/completion boundary")
    if len(ids) > max_length:
        raise ValueError(f"{row['id']}: {len(ids)} tokens exceeds {max_length}; no truncation")
    target = ids[boundary:]
    if not target or target[-1] != tokenizer.eos_token_id or target.count(tokenizer.eos_token_id) != 1:
        raise ValueError("LFM completion must end at its single true EOS")
    return {"id": row["id"], "input_ids": ids, "labels": [-100] * boundary + target,
            "prefix_ids": prefix_ids, "boundary": boundary, "target_tokens": len(target),
            "prompt_bytes": len(row["prompt"].encode()), "completion_bytes": len(row["completion"].encode())}


def set_training_mode(model):
    base = _base_model(model)
    model.train()
    for component in (base.model.vision_tower, base.model.multi_modal_projector):
        if any(parameter.requires_grad for parameter in component.parameters()):
            raise ValueError("LFM vision/connector is trainable")
        component.eval()


def _validate_recipe(model):
    if not hasattr(model, "peft_config") or set(model.peft_config) != {"default"}:
        raise ValueError("LFM requires exactly one PEFT default adapter")
    config = model.peft_config["default"]
    required = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 8,
        "lora_alpha": 16, "lora_dropout": 0.05, "bias": "none",
        "inference_mode": False, "use_rslora": False, "use_dora": False,
        "fan_in_fan_out": False, "lora_bias": False, "use_qalora": False,
        "modules_to_save": None, "target_parameters": None,
        "trainable_token_indices": None, "layers_to_transform": None,
        "layers_pattern": None, "rank_pattern": {}, "alpha_pattern": {},
        "layer_replication": None, "alora_invocation_tokens": None,
        "loftq_config": {}, "eva_config": None, "corda_config": None,
        "init_lora_weights": True, "exclude_modules": None,
        "megatron_config": None, "lora_ga_config": None, "velora_config": None,
        "monteclora_config": None, "use_bdlora": None, "arrow_config": None,
        "kasa_config": None, "ensure_weight_tying": False,
    }
    for name, value in required.items():
        if getattr(config, name, None) != value:
            raise ValueError("unqualified LFM LoRA recipe: " + name)
    if config.target_modules != PROFILE["lora_targets"]:
        raise ValueError("LFM PEFT target configuration changed")
    if list(model.active_adapters) != ["default"]:
        raise ValueError("wrong active LFM adapter")


def _validate_trainable(model, selected, *, expected_device):
    """Shared inventory implementation; meta is permitted only in private tests."""
    _selected(selected)
    _validate_recipe(model)
    import torch
    base = _base_model(model)
    parameters = dict(base.named_parameters())
    expected = {name + ".lora_" + side + ".default.weight"
                for name in PROFILE["lora_target_paths"] for side in ("A", "B")}
    actual = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if actual != expected:
        extra, missing = sorted(actual - expected), sorted(expected - actual)
        raise ValueError(f"LFM trainable allowlist mismatch; extra={extra}; missing={missing}")
    for name, parameter in parameters.items():
        if parameter.device.type != expected_device or parameter.dtype != torch.float32:
            raise ValueError("LFM parameter device/dtype mismatch: " + name)
    adapters = {name for name, module in base.named_modules() if hasattr(module, "lora_A")}
    if adapters != set(PROFILE["lora_target_paths"]):
        raise ValueError("LFM contains adapters outside exact language whitelist")
    for name in adapters:
        module = base.get_submodule(name)
        if (set(module.lora_A) != {"default"} or set(module.lora_B) != {"default"}
                or module.lora_A["default"].weight.shape[0] != 8
                or module.lora_B["default"].weight.shape[1] != 8
                or module.r != {"default": 8} or module.lora_alpha != {"default": 16}
                or module.scaling != {"default": 2.0}
                or module.lora_dropout["default"].p != 0.05
                or module.merged_adapters):
            raise ValueError("LFM requires one rank-8 language adapter: " + name)
    expected_inventory = PROFILE["expected_inventory"]
    vision, connector = base.model.vision_tower, base.model.multi_modal_projector
    for component, label in ((vision, "vision"), (connector, "connector")):
        if (sum(parameter.numel() for parameter in component.parameters()) != expected_inventory[label + "_parameters"]
                or any(parameter.requires_grad for parameter in component.parameters())
                or component.training):
            raise ValueError("LFM frozen component inventory changed: " + label)
    trainable_count = sum(parameters[name].numel() for name in actual)
    frozen_count = sum(parameter.numel() for parameter in parameters.values() if not parameter.requires_grad)
    if (trainable_count != expected_inventory["lora_parameters"]
            or frozen_count != expected_inventory["base_parameters"]):
        raise ValueError("LFM trainable/frozen parameter totals changed")
    return {
        "model_id": MODEL_ID, "revision": REVISION, "model_class": type(base).__name__,
        "framing": FRAMING_VERSION, "device": expected_device,
        "meta_only_non_training_check": expected_device == "meta", "dtype": "float32",
        "trainable_parameters": trainable_count, "trainable_tensors": len(actual),
        "language_lora_modules": len(adapters), "frozen_base_parameters": frozen_count,
        "vision_class": type(vision).__name__, "frozen_vision_parameters": expected_inventory["vision_parameters"],
        "connector_class": type(connector).__name__, "frozen_connector_parameters": expected_inventory["connector_parameters"],
        "vision_training": vision.training, "connector_training": connector.training,
        "frozen_weight_sha256": PROFILE["required_hashes"]["model.safetensors"],
        "frozen_byte_verification": "common trainer must compare every frozen tensor before and after training",
        "kernel_backend": "stock-torch-cpu-conv1d", "attention": "eager",
        "lora_targets": sorted(adapters),
    }


def validate_trainable(model, selected):
    """Public training gate: real CPU FP32 parameters and the exact LoRA recipe."""
    return _validate_trainable(model, selected, expected_device="cpu")
