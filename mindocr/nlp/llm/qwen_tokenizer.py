import base64
import unicodedata
from typing import Collection, Dict, List, Set, Union

import numpy as np
import tiktoken

import mindspore as ms

PAT_STR = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+
(?!\S)|\s+"""
ENDOFTEXT = "<|endoftext|>"
IMSTART = "<|im_start|>"
IMEND = "<|im_end|>"
EXTRAS = tuple((f"<|extra_{i}|>" for i in range(205)))
SPECIAL_TOKENS = (
    (
        ENDOFTEXT,
        IMSTART,
        IMEND,
    )
    + EXTRAS
    + ("<ref>", "</ref>", "<box>", "</box>", "<quad>", "</quad>", "<img>", "</img>", "<imgpad>")
)


def to_py_obj(obj):
    """
    Convert a Mindspore tensor, Numpy array or python list to a python list.
    """
    if isinstance(obj, dict):
        return {k: to_py_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py_obj(o) for o in obj]
    if isinstance(obj, ms.Tensor):
        return obj.asnumpy().tolist()
    if isinstance(obj, (np.ndarray, np.number)):  # tolist also works on 0d np arrays
        return obj.tolist()
    return obj


def _load_tiktoken_bpe(tiktoken_bpe_file: str) -> Dict[bytes, int]:
    with open(tiktoken_bpe_file, "rb") as f:
        contents = f.read()
    return {
        base64.b64decode(token): int(rank) for token, rank in (line.split() for line in contents.splitlines() if line)
    }


class QwenTokenizer:
    """Qwen Tokenizer"""

    def __init__(self, vocab_file="qwen.tiktoken", pad_token=ENDOFTEXT):
        self.vocab_file = vocab_file
        self.mergeable_ranks = _load_tiktoken_bpe(vocab_file)

        self.special_tokens = {
            token: index for index, token in enumerate(SPECIAL_TOKENS, start=len(self.mergeable_ranks))
        }

        enc = tiktoken.Encoding(
            "Qwen",
            pat_str=PAT_STR,
            mergeable_ranks=self.mergeable_ranks,
            special_tokens=self.special_tokens,
        )
        assert (
            len(self.mergeable_ranks) + len(self.special_tokens) == enc.n_vocab
        ), f"{len(self.mergeable_ranks) + len(self.special_tokens)} != {enc.n_vocab} in encoding"

        self.decoder = {v: k for k, v in self.mergeable_ranks.items()}
        self.decoder.update({v: k for k, v in self.special_tokens.items()})

        self.tokenizer = enc

        self.eod_id = self.tokenizer.eot_token
        self.im_start_id = self.special_tokens[IMSTART]
        self.im_end_id = self.special_tokens[IMEND]

        self.errors = "replace"
        self._in_target_context_manager = False
        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.pad_token_type_id = 0
        self.pad_token_id = self.convert_tokens_to_ids(pad_token)

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab

    # override Tokenizer.convert_tokens_to_string()
    def convert_tokens_to_string(self, tokens: List[Union[bytes, str]]) -> str:
        """
        Converts a sequence of tokens in a single string.
        """
        text = ""
        temp = b""
        for t in tokens:
            if isinstance(t, str):
                if temp:
                    text += temp.decode("utf-8", errors=self.errors)
                    temp = b""
                text += t
            elif isinstance(t, bytes):
                temp += t
            else:
                raise TypeError("token should only be of type types or str")
        if temp:
            text += temp.decode("utf-8", errors=self.errors)
        return text

    # called by Tokenizer.convert_tokens_to_ids() & SpecialTokensMixin
    def _convert_tokens_to_ids(self, tokens: Union[bytes, str, List[Union[bytes, str]]]) -> Union[int, List[int]]:
        """Convert the tokens to ids using vocab mapping"""
        if isinstance(tokens, (str, bytes)):
            return self._convert_token_to_id(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token: Union[bytes, str]) -> int:
        """Converts a token to an id using the vocab, special tokens included"""
        if token in self.special_tokens:
            return self.special_tokens[token]
        if token in self.mergeable_ranks:
            return self.mergeable_ranks[token]
        raise ValueError("unknown token")

    # required by Tokenizer.convert_ids_to_tokens() of mindformers<=0.6
    def _convert_ids_to_tokens(self, input_id: int):
        return self._convert_id_to_token(input_id)

    # called by Tokenizer.convert_ids_to_tokens()
    def _convert_id_to_token(self, index: int) -> Union[bytes, str]:
        """Converts an id to a token, special tokens included"""
        if index in self.decoder:
            return self.decoder[index]
        raise ValueError("unknown ids")

    def tokenize(
        self,
        text: str,
        allowed_special: Union[Set, str] = "all",
        disallowed_special: Union[Collection, str] = (),
    ) -> List[Union[bytes, str]]:
        """
        Converts a string in a sequence of tokens.

        Args:
            text (`str`):
                The sequence to be encoded.
            allowed_special (`Literal["all"]` or `set`):
                The surface forms of the tokens to be encoded as special tokens in regular texts.
                Default to "all".
            disallowed_special (`Literal["all"]` or `Collection`):
                The surface forms of the tokens that should not be in regular texts and trigger errors.
                Default to an empty tuple.

        Returns:
            `List[bytes|str]`: The list of tokens.
        """
        tokens = []
        text = unicodedata.normalize("NFC", text)

        # this implementation takes a detour: text -> token id -> token surface forms
        for t in self.tokenizer.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special):
            tokens.append(self.decoder[t])
        return tokens

    def _decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        """override Tokenizer._decode(), called by BaseTokenizer.decode()"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            token_ids = [i for i in token_ids if i < self.eod_id]
        return self.tokenizer.decode(token_ids, errors=self.errors)

    def _call_one(self, text, max_length=None):
        is_batched = isinstance(text, (list, tuple))

        if is_batched:
            return self.batch_encode_plus(batch_text_or_text_pairs=text, max_length=max_length)
        outputs = self.batch_encode_plus(batch_text_or_text_pairs=[text], max_length=max_length)
        outputs = {_: outputs[_][0] for _ in outputs}
        return outputs

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int], None]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]` or `None`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None
        return self._convert_token_to_id(token)

    def batch_encode_plus(self, batch_text_or_text_pairs, max_length):
        input_ids = []
        for ids in batch_text_or_text_pairs:
            tokens = self.tokenize(ids)
            input_id = self.convert_tokens_to_ids(tokens)
            input_ids.append(input_id)

        batch_outputs = {}
        for input_id in input_ids:
            outputs = self.prepare_for_model(input_id)

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        batch_outputs = self.pad(batch_outputs, max_length=max_length)

        return batch_outputs

    def pad(self, encoded_inputs, max_length=None):
        # The model's main input name, usually `input_ids`, has be passed for padding
        if self.model_input_names[0] not in encoded_inputs:
            raise ValueError(
                "You should supply an encoding or a list of encodings to this method "
                f"that includes {self.model_input_names[0]}, but you provided {list(encoded_inputs.keys())}"
            )

        required_input = encoded_inputs[self.model_input_names[0]]

        batch_size = len(required_input)

        batch_outputs = {}
        for i in range(batch_size):
            inputs = {k: v[i] for k, v in encoded_inputs.items()}

            outputs = self._pad(inputs, max_length=max_length)
            outputs = {_: outputs[_][:max_length] for _ in outputs}

            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return batch_outputs

    def _pad(self, encoded_inputs, max_length) -> dict:
        return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]
        required_input_len = len(required_input)

        needs_to_be_padded = required_input_len < max_length

        # Initialize attention mask if not present.
        if return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * required_input_len

        if needs_to_be_padded:
            difference = max_length - required_input_len

            if return_attention_mask:
                encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
            if "token_type_ids" in encoded_inputs:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                )
            if "special_tokens_mask" in encoded_inputs:
                encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
            encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference

        return encoded_inputs

    def prepare_for_model(self, ids):
        return_token_type_ids = "token_type_ids" in self.model_input_names
        encoded_inputs = {"input_ids": ids}
        if return_token_type_ids:
            encoded_inputs["token_type_ids"] = [0] * len(ids)
        return encoded_inputs

    def __call__(self, text, max_length):
        """
        Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
        sequences.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`, *optional*):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            max_length (int): Max sequence length.
        """
        encodings = self._call_one(text=text, max_length=max_length)
        return encodings

    def decode(self, token_ids, skip_special_tokens=False) -> str:
        # Convert inputs to python lists
        token_ids = to_py_obj(token_ids)

        if isinstance(token_ids[0], list):
            output = []
            for item in token_ids:
                new_strs = self._decode(token_ids=item, skip_special_tokens=skip_special_tokens)
                output.append(new_strs)
        else:
            output = self._decode(token_ids=token_ids, skip_special_tokens=skip_special_tokens)
        return output
