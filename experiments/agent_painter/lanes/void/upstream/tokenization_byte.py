import json
import os
from transformers import PreTrainedTokenizer


BYTE_PREFIX = "<0x"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


class ByteTokenizer(PreTrainedTokenizer):
    """Lossless UTF-8 byte tokenizer used by Qwen90M.

    IDs:
      0..255 -> raw byte values
      256    -> PAD
      257    -> BOS
      258    -> EOS

    No UNK token is required because every UTF-8 string is representable as bytes.
    """

    vocab_files_names = {"vocab_file": "byte_vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        pad_token=PAD_TOKEN,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        unk_token=None,
        model_max_length=1024,
        padding_side="left",
        clean_up_tokenization_spaces=False,
        **kwargs,
    ):
        # Transformers v5 loads values from tokenizer_config.json into this
        # constructor. Make every value that we also forward to PythonBackend
        # an explicit argument so it is consumed exactly once instead of being
        # duplicated inside **kwargs.
        self.vocab_file = vocab_file
        kwargs.setdefault("split_special_tokens",True)
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            model_max_length=model_max_length,
            padding_side=padding_side,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return 259

    def get_vocab(self):
        vocab = {f"<0x{i:02X}>": i for i in range(256)}
        vocab[PAD_TOKEN] = 256
        vocab[BOS_TOKEN] = 257
        vocab[EOS_TOKEN] = 258
        return vocab

    def _tokenize(self, text, **kwargs):
        return [f"<0x{b:02X}>" for b in text.encode("utf-8", errors="replace")]

    def _convert_token_to_id(self, token):
        if token == PAD_TOKEN:
            return 256
        if token == BOS_TOKEN:
            return 257
        if token == EOS_TOKEN:
            return 258
        if isinstance(token, str) and token.startswith(BYTE_PREFIX) and token.endswith(">"):
            try:
                value = int(token[3:-1], 16)
                if 0 <= value <= 255:
                    return value
            except ValueError:
                pass
        # This branch should be unreachable for text encoded by this tokenizer.
        return 0

    def _convert_id_to_token(self, index):
        index = int(index)
        if 0 <= index <= 255:
            return f"<0x{index:02X}>"
        if index == 256:
            return PAD_TOKEN
        if index == 257:
            return BOS_TOKEN
        if index == 258:
            return EOS_TOKEN
        return "<0x00>"

    def convert_tokens_to_string(self, tokens):
        out = []
        buf = bytearray()

        def flush():
            nonlocal buf
            if buf:
                out.append(bytes(buf).decode("utf-8", errors="replace"))
                buf = bytearray()

        for token in tokens:
            idx = self._convert_token_to_id(token)
            if isinstance(token, str) and 0 <= idx <= 255 and token.startswith(BYTE_PREFIX):
                buf.append(idx)
            else:
                flush()
                out.append(str(token))
        flush()
        return "".join(out)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # BET pretraining did not automatically insert BOS/EOS around ordinary text.
        if token_ids_1 is None:
            return list(token_ids_0)
        return list(token_ids_0) + list(token_ids_1)

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        n = len(token_ids_0) + (len(token_ids_1) if token_ids_1 is not None else 0)
        return [0] * n

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        name = "byte_vocab.json" if filename_prefix is None else f"{filename_prefix}-byte_vocab.json"
        path = os.path.join(save_directory, name)
        vocab = {f"<0x{i:02X}>": i for i in range(256)}
        vocab.update({PAD_TOKEN: 256, BOS_TOKEN: 257, EOS_TOKEN: 258})
        with open(path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2, sort_keys=True)
        return (path,)

