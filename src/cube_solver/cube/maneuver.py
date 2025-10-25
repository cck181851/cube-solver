"""Maneuver module."""
from __future__ import annotations

import random
from collections import Counter
from itertools import combinations
from typing import Union, List, Tuple, Iterator, overload

from .cube import Cube
from .enums import Move
from ..defs import NEXT_MOVES

moves = [Move.NONE] + [*Move.moves()]
REDUCED_MOVES = {frozenset(): []}
for first, second in combinations(moves, 2):
    if first.axis == second.axis and first.name[:-1] != second.name[:-1]:
        counter = Counter(dict(zip(first.layers, first.shifts)))
        counter.update(dict(zip(second.layers, second.shifts)))
        key = frozenset((layer, shift % 4) for layer, shift in counter.items() if shift % 4)
        if key not in REDUCED_MOVES:
            REDUCED_MOVES[key] = [first, second]
    elif first == Move.NONE:
        counter = Counter(dict(zip(second.layers, second.shifts)))
        key = frozenset((layer, shift % 4) for layer, shift in counter.items())
        REDUCED_MOVES[key] = [second]


def _reduce_moves(moves: List[Move]) -> List[Move]:
    """
    Given a sequence of moves, reduce consecutive
    moves along the same axis where possible.
    """
    for i in range(len(moves) - 1):
        if moves[i].axis == moves[i+1].axis:
            counter = Counter(dict(zip(moves[i].layers, moves[i].shifts)))
            reduced_moves = _reduced(counter, moves, i, 1)
            if reduced_moves is not None:
                return reduced_moves
    return moves


def _reduced(counter: Counter, moves: List[Move], i: int, n: int) -> Union[List[Move],  None]:
    """
    Update counter and reduce consecutive moves along the same axis where possible.
    Returns ``None`` if the sequence of consecutive moves starting at index ``i`` cannot be reduced.
    """
    counter.update(dict(zip(moves[i+n].layers, moves[i+n].shifts)))
    key = frozenset((layer, shift % 4) for layer, shift in counter.items() if shift % 4)
    if key in REDUCED_MOVES and len(REDUCED_MOVES[key]) <= n:
        return _reduce_moves(moves[:i] + REDUCED_MOVES[key] + moves[i+n+1:])
    if i + n + 1 < len(moves) and moves[i].axis == moves[i+n+1].axis:
        return _reduced(counter, moves, i, n + 1)


class Maneuver(str):
    def __new__(cls, moves: Union[str, List[Move]], reduce: bool = True):
        """
        Create :class:`Maneuver` object.
        A :class:`Maneuver` is a subclass of :class:`str` that represents
        the sequence of moves that can be applied to a :class:`cube_solver.Cube` object.
        """
        cls.moves: Tuple[Move, ...]

        if not isinstance(moves, (str, list)):
            raise TypeError(f"moves must be str or list, not {type(moves).__name__}")
        if not isinstance(reduce, bool):
            raise TypeError(f"reduce must be bool, not {type(reduce).__name__}")

        if isinstance(moves, str):
            moves = [Move.from_string(move_str) for move_str in moves.split()]
        else:
            for move in moves:
                if not isinstance(move, Move):
                    raise TypeError(f"moves list elements must be Move, not {type(move).__name__}")

        if reduce:
            moves = _reduce_moves([move for move in moves if move != Move.NONE])
        obj = super().__new__(cls, " ".join(move.string for move in moves))
        obj.moves = tuple(moves)
        return obj

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Maneuver):
            try:
                if not isinstance(other, (str, list)):
                    return False
                other = Maneuver(other)
            except Exception:
                return False
        return repr(Cube(self)) == repr(Cube(other))

    def __len__(self) -> int:
        return len(self.moves)

    @overload
    def __getitem__(self, key: int) -> Move: ...
    @overload
    def __getitem__(self, key: slice) -> Maneuver: ...

    def __getitem__(self, key: Union[int, slice]) -> Union[Move, Maneuver]:  # type: ignore
        if not isinstance(key, (int, slice)):
            raise TypeError(f"Maneuver indices must be int or slice, not {type(key).__name__}")
        if isinstance(key, int):
            try:
                return self.moves[key]
            except IndexError:
                raise IndexError("Maneuver index out of range")
        return Maneuver([*self.moves[key]])

    def __iter__(self) -> Iterator[Move]:  # type: ignore
        for move in self.moves:
            yield move

    def __contains__(self, item: Union[str, Move]) -> bool:
        if isinstance(item, str):
            item = Move.from_string(item)
        return item in self.moves

    def __neg__(self) -> Maneuver:
        return self.inverse

    def __add__(self, other: Union[str, List[Move]]) -> Maneuver:
        if not isinstance(other, Maneuver):
            other = Maneuver(other)
        return Maneuver([*self.moves] + [*other.moves])

    def __radd__(self, other: Union[str, List[Move]]) -> Maneuver:
        other = Maneuver(other)
        return other.__add__(self)

    def __sub__(self, other: Union[str, List[Move]]) -> Maneuver:
        if not isinstance(other, Maneuver):
            other = Maneuver(other)
        return self.__add__(other.__neg__())

    def __rsub__(self, other: Union[str, List[Move]]) -> Maneuver:
        other = Maneuver(other)
        return other.__sub__(self)

    def __mul__(self, other: Union[int, str, List[Move]]) -> Maneuver:  # type: ignore
        if isinstance(other, int):
            return Maneuver([*self.moves] * other)
        if not isinstance(other, Maneuver):
            other = Maneuver(other)
        return self.__add__(other).__sub__(self)

    def __rmul__(self, other: Union[int, str, List[Move]]) -> Maneuver:  # type: ignore
        if isinstance(other, int):
            return Maneuver([*self.moves] * other)
        other = Maneuver(other)
        return other.__mul__(self)

    def __matmul__(self, other: Union[str, List[Move]]) -> Maneuver:
        if not isinstance(other, Maneuver):
            other = Maneuver(other)
        return self.__mul__(other).__sub__(other)

    def __rmatmul__(self, other: Union[str, List[Move]]) -> Maneuver:
        other = Maneuver(other)
        return other.__matmul__(self)

    @property
    def inverse(self) -> Maneuver:
        """Inverse maneuver."""
        return Maneuver([move.inverse for move in self.moves[::-1]])

    @classmethod
    def random(cls, length: int = 25) -> Maneuver:
        """Generate a random maneuver."""
        if not isinstance(length, int):
            raise TypeError(f"length must be int, not {type(length).__name__}")
        if length < 0:
            raise ValueError(f"length must be >= 0 (got {length})")

        moves = [Move.NONE]
        for i in range(length):
            moves.append(random.choice(NEXT_MOVES[moves[-1]]))
        return cls(moves[1:])
