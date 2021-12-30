from typing import List, Iterable

from src.data.objects.stack import Stack
from src.preprocess.entry_coders import Entry2Seq
from src.preprocess.token import Token
from src.preprocess.tokenizers import Padding, Tokenizer


class VocabFreqController:
    def __init__(self, min_freq: int = 0, oov: str = "OOV"):
        self._min_freq = min_freq
        self._freqs = {}
        self._frequent_words = set()
        self._oov = oov

    def fit(self, texts: Iterable[List[Token[str]]]) -> "VocabFreqController":
        for text in texts:
            for token in text:
                self._freqs[token.value] = self._freqs.get(token.value, 0) + 1
        for word in self._freqs:
            if self._freqs[word] >= self._min_freq:
                self._frequent_words.add(word)
        return self

    def encode(self, text: List[Token[str]]) -> List[Token[str]]:
        if self._min_freq <= 0:
            return text
        return [Token(token.value if token.value in self._frequent_words else self._oov, token.data)
                for token in text]

    def __call__(self, text: List[Token[str]]) -> List[Token[str]]:
        return self.encode(text)

    def transform(self, texts: Iterable[List[Token[str]]]) -> List[List[Token[str]]]:
        return [self.encode(text) for text in texts]

    def fit_transform(self, texts: Iterable[List[Token[str]]]) -> List[List[Token[str]]]:
        return self.fit(texts).transform(texts)

    def __len__(self) -> int:
        return len(self._frequent_words) + 1

    def __str__(self) -> str:
        return "oov" + str(self._min_freq)


class CharFilter:
    def __init__(self):
        self._ok_symbols = set([chr(i) for i in range(ord("a"), ord("z") + 1)]
                               + [".", ","] + [chr(i) for i in range(ord("0"), ord("9") + 1)])  # $

    def __call__(self, seq: List[Token[str]]) -> List[Token[str]]:
        filtered_seq = []
        for token in seq:
            value = "".join(filter(lambda x: x.lower() in self._ok_symbols, token.value))
            if value:
                filtered_seq.append(Token(value, token.data))

        return filtered_seq


class SeqCoder:
    def __init__(self, entry_to_seq: Entry2Seq, tokenizer: Tokenizer, min_freq: int = 0, max_len: int = None):
        self._entry_to_seq = entry_to_seq
        self._char_filter = CharFilter()
        self._vocab_control = VocabFreqController(min_freq)
        self._tokenizer = Padding(tokenizer, max_len)
        self._fitted = False
        self._name = "_".join(
            filter(lambda x: x.strip(),
                   (str(self._entry_to_seq), str(self._tokenizer), str(self._vocab_control))))

    def fit(self, stacks: List[Stack]) -> "SeqCoder":
        if self._fitted:
            print("SeqCoder already fitted, fit call skipped")
            return self
        seqs = [self._char_filter(self._entry_to_seq(stack)) for stack in stacks]
        if self._vocab_control:
            seqs = self._vocab_control.fit_transform(seqs)
        self._tokenizer.fit(seqs)
        self._fitted = True
        return self

    def _pre_call(self, stack: Stack) -> List[Token[str]]:
        res = stack
        for tr in [self._entry_to_seq, self._char_filter, self._vocab_control]:
            if tr is not None:
                res = tr(res)
        return res

    def transform(self, stack: Stack) -> List[Token[int]]:
        return self._tokenizer(self._pre_call(stack))

    def transforms(self, stacks: List[Stack]) -> List[List[Token[int]]]:
        return [self.transform(stack) for stack in stacks]

    def to_seq(self, stack: Stack) -> List[Token[str]]:
        return self._tokenizer.split(self._pre_call(stack))

    def __len__(self) -> int:
        return len(self._tokenizer)

    def __str__(self) -> str:
        return self._name

    def train(self, mode: bool = True):
        self._tokenizer.train(mode)
