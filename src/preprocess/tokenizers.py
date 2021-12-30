from abc import ABC, abstractmethod
from typing import List, Union, Any

from src.data.objects.frame import FrameData
from src.preprocess.token import Token


class IdCoder:
    def __init__(self):
        self._unk_symbol = "_|_"
        self._id2name = {0: self._unk_symbol}
        self._name2id = {self._unk_symbol: 0}
        self._fixed = False

    def encode(self, word: Any) -> Union[int, None]:
        if not self._fixed and (word not in self._name2id):
            self._name2id[word] = len(self._name2id)
            self._id2name[self._name2id[word]] = word
        return self._name2id.get(word, 0)

    def __getitem__(self, item: Any) -> Union[int, None]:
        return self.encode(item)

    def encodes(self, words: List[Any]) -> List[Union[int, None]]:
        return [self.encode(word) for word in words]

    def decode(self, id: int) -> Any:
        return self._id2name[id]

    def decodes(self, ids: List[int]) -> List[Any]:
        return [self.decode(id) for id in ids]

    def fix(self):
        self._fixed = True

    def __len__(self) -> int:
        return len(self._name2id)


class Tokenizer(ABC):
    def __init__(self):
        self._train = False

    @abstractmethod
    def fit(self, texts: List[List[Token[str]]]) -> "Tokenizer":
        return self

    @abstractmethod
    def encode(self, text: List[Token[str]]) -> List[Token[int]]:
        raise NotImplementedError

    @abstractmethod
    def split(self, text: List[Token[str]]) -> List[Token[str]]:
        raise NotImplementedError

    def __call__(self, text: List[Token[str]],
                 type: str = "id") -> Union[List[Token[str]], List[Token[int]]]:
        if type == 'id':
            return self.encode(text)
        else:
            return self.split(text)

    @abstractmethod
    def to_str(self, id: Token[int]) -> Token[str]:
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def train(self, mode: bool = True):
        self._train = mode


class Padding(Tokenizer):
    def __init__(self, tokenizer: "Tokenizer", max_len: int = None, pad: str = "[PAD]"):
        super().__init__()
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._pad = pad
        self._offset = 1
        self._pad_id = 0

    def fit(self, texts: List[List[Token[str]]]) -> "Padding":
        self._tokenizer.fit(texts)
        return self

    def pad_seq(self, seq: List[Any], pad: Any) -> List[Any]:
        if self._max_len is not None:
            if len(seq) < self._max_len:
                return [pad] * (self._max_len - len(seq)) + seq
            else:
                return seq[len(seq) - min(len(seq), self._max_len):]
        return seq

    def encode(self, text: List[Token[str]]) -> List[Token[int]]:
        encoded_seq = self._tokenizer.encode(text)
        encoded_seq = [Token(token.value + self._offset, token.data)
                       for token in encoded_seq]
        return self.pad_seq(encoded_seq, Token(self._pad_id, FrameData()))

    def split(self, text: List[Token[str]]) -> List[Token[str]]:
        return self.pad_seq(self._tokenizer.split(text), Token(self._pad, FrameData()))

    def to_str(self, id: Token[int]) -> Token[str]:
        if id.value >= self._offset:
            return self._tokenizer.to_str(Token(id.value - self._offset, id.data))
        elif id.value == self._pad_id:
            return Token(self._pad, id.data)
        else:
            raise ValueError("Unknown token id")

    def __len__(self) -> int:
        return len(self._tokenizer) + self._offset

    def __str__(self) -> str:
        return str(self._tokenizer) + (f"_pad{self._max_len}" if self._max_len is not None else "")

    def train(self, mode: bool = True):
        super().train(mode)
        self._tokenizer.train(mode)


class SimpleTokenizer(Tokenizer):
    def __init__(self, name: str):
        super().__init__()
        self._coder = IdCoder()
        self._name = name

    @abstractmethod
    def to_strs(self, frames: List[Token[str]]) -> List[Token[str]]:
        raise NotImplementedError

    def fit(self, texts: List[List[Token[str]]]) -> "SimpleTokenizer":
        for text in texts:
            for seq in self.to_strs(text):
                self._coder.encode(seq.value)
        self._coder.fix()
        return self

    def encode(self, text: List[Token[str]]) -> List[Token[int]]:
        codes = []
        for seq in self.to_strs(text):
            code = self._coder.encode(seq.value)
            if code is not None:
                codes.append(Token(code, seq.data))
        return codes

    def split(self, text: List[Token[str]]) -> List[Token[str]]:
        return self.to_strs(text)

    def to_str(self, id: Token[int]) -> Token[str]:
        return Token(self._coder.decode(id.value), id.data)

    def __len__(self):
        return len(self._coder)

    def __str__(self) -> str:
        return self._name


class FrameTokenizer(SimpleTokenizer):
    def __init__(self):
        super().__init__("frame")
        self._coder = IdCoder()

    def to_strs(self, frames: List[Token[str]]) -> List[Token[str]]:
        return frames
