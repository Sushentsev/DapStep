from abc import ABC, abstractmethod
from typing import List

from src.data.objects.stack import Stack
from src.preprocess.token import Token


class Entry2Seq(ABC):
    @abstractmethod
    def __call__(self, stack: Stack) -> List[Token[str]]:
        pass


def remove_equals(tokens: List[Token[str]]) -> List[Token[str]]:
    res = []
    for i, t in enumerate(tokens):
        if (i == 0 or tokens[i - 1].value != t.value) and (t.value.strip() != ""):
            res.append(t)
    return res


class Entry2SeqHelper:
    def __init__(self, cased: bool = True, trim_len: int = 0, rem_equals: bool = False):
        self._cased = cased
        self._trim_len = trim_len
        self._rem_equals = rem_equals
        self._name = ("" if cased else "un") + "cs" + (f"_tr{trim_len}" if trim_len > 0 else "")

    def __call__(self, seq: List[Token[str]]) -> List[Token[str]]:
        if self._trim_len > 0:
            seq = [Token(".".join(token.value.split(".")[:-self._trim_len]), token.data)
                   for token in seq]

        if not self._cased:
            seq = [Token(token.value.lower(), token.data) for token in seq]

        if self._rem_equals:
            seq = remove_equals(seq)

        return seq

    def __str__(self) -> str:
        return self._name


class Stack2FrameNames(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int = 0, rem_equals: bool = False):
        self._helper = Entry2SeqHelper(cased, trim_len, rem_equals)

    def __call__(self, stack: Stack) -> List[Token[str]]:
        return self._helper([Token(frame.raw_frame.name, frame.data)
                             for frame in stack.frames])

    def __str__(self) -> str:
        return "s2fn_" + str(self._helper)


class Stack2FrameFileNames(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int = 0, rem_equals: bool = False):
        self._helper = Entry2SeqHelper(cased, trim_len, rem_equals)

    def __call__(self, stack: Stack) -> List[Token[str]]:
        return self._helper([Token(frame.raw_frame.file_name, frame.data)
                             for frame in stack.frames
                             if frame.raw_frame.file_name])  # Some file_names might be null.

    def __str__(self) -> str:
        return "s2ffn_" + str(self._helper)


class Stack2FrameSubsystems(Entry2Seq):
    def __init__(self, cased: bool = True, trim_len: int = 0, rem_equals: bool = False):
        self._helper = Entry2SeqHelper(cased, trim_len, rem_equals)

    def __call__(self, stack: Stack) -> List[Token[str]]:
        return self._helper([Token(frame.raw_frame.subsystem, frame.data)
                             for frame in stack.frames
                             if frame.raw_frame.subsystem])

    def __str__(self) -> str:
        return "s2fs_" + str(self._helper)
