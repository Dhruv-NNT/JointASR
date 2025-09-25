"""Character tokenizer and loader utilities."""

import json
from typing import Dict, List, Union

class CharTokenizer:
    """
    Very small, file-backed character tokenizer with:
      - <pad> (also used as CTC blank)
      - <s> (SOS), </s> (EOS)
      - '|' used as the space token
    """

    def __init__(self, vocab: Union[Dict[str, int], List[str]]):
        # Accept either dict {token: id} or list [token0, token1, ...]
        if isinstance(vocab, dict):
            self.stoi: Dict[str, int] = dict(vocab)
            # Invert to itos (dense ids 0..V-1 expected)
            self.itos: Dict[int, str] = {i: t for t, i in self.stoi.items()}
            # Build dense token list if possible
            self.tokens: List[str] = [self.itos[i] for i in range(len(self.itos))]
        else:
            self.tokens = list(vocab)
            self.stoi = {t: i for i, t in enumerate(self.tokens)}
            self.itos = {i: t for i, t in enumerate(self.tokens)}

        # Special tokens (defaults if not present)
        self.pad_id  = self.stoi.get("<pad>", self.stoi.get("[PAD]", 0))
        self.blank_id = self.pad_id  # CTC blank == pad by convention

        self.unk_id  = self.stoi.get("<unk>", self.stoi.get("[UNK]", 1))
        self.sos_id  = self.stoi.get("<s>",   self.stoi.get("[SOS]", 2))
        self.eos_id  = self.stoi.get("</s>",  self.stoi.get("[EOS]", 3))

        # Space token used by text<->ids conversion
        self.space_token = "|" if "|" in self.stoi else " "  # fallback if needed

        # Detect whether vocab is uppercase or lowercase (heuristic)
        # If any alphabetic token is uppercase, we treat vocab as uppercase.
        self._vocab_is_upper = any(t.isalpha() and t.upper() == t for t in self.tokens)
        # --- Enforce strict ids like the reference ---
        # <pad>=0, blank==pad, <s>=1, </s>=2, '|' must exist
        assert self.pad_id == 0, "Expected <pad> id 0"
        assert self.blank_id == self.pad_id, "CTC expects blank==pad"
        assert self.sos_id == 1 and self.eos_id == 2, "Expected <s>=1 and </s>=2"
        assert "|" in self.stoi, "Space token '|' must exist in vocab"

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """
        Load a tokenizer from a JSON file.
        Accepts either:
           - {"tokens": [...]} or
           - [..., ...] or
           - {"<pad>": 0, "a": 1, ...}
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "tokens" in data:
            vocab = data["tokens"]
        else:
            vocab = data
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def text2ids(self, text: str, add_sos_eos: bool = False):
        """
        Map raw text -> tensor of ids.
        - Normalizes to lower/upper to match vocab style.
        - Replaces spaces with the '|' space token.
        """
        text = (text or "").strip()
        if self._vocab_is_upper:
            norm = text.upper()
        else:
            norm = text.lower()
        norm = norm.replace(" ", self.space_token)

        ids = [self.stoi.get(ch, self.unk_id) for ch in norm]
        if add_sos_eos:
            ids = [self.sos_id] + ids + [self.eos_id]

        import torch
        return torch.tensor(ids, dtype=torch.long)

    def ids2text(self, ids) -> str:
        """
        Map a sequence of ids -> string (spaces restored from '|').
        Accepts list/tuple/Tensor.
        """
        import torch
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        s = "".join(self.itos.get(int(i), "") for i in ids)
        return s.replace(self.space_token, " ")

# Optional helper to keep old naming in other modules
def build_tokenizer(vocab_json: str) -> CharTokenizer:
    return CharTokenizer.load(vocab_json)
