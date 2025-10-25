"""Cube utils module."""
import math
import numpy as np
from typing import Tuple

from .defs import FACTORIAL, COMBINATION


def get_orientation_array(coord: int, v: int, n: int, force_modulo: bool = False) -> np.ndarray:
    """Get orientation array."""
    if not isinstance(coord, int):
        raise TypeError(f"coord must be int, not {type(coord).__name__}")
    if not isinstance(v, int):
        raise TypeError(f"v must be int, not {type(v).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, not {type(n).__name__}")
    if not isinstance(force_modulo, bool):
        raise TypeError(f"force_modulo must be bool, not {type(force_modulo).__name__}")
    if v <= 0:
        raise ValueError(f"v must be positive (got {v})")
    if n <= 0:
        raise ValueError(f"n must be positive (got {n})")
    upper_lim = v ** n
    if force_modulo:
        upper_lim //= v
    if coord < 0 or coord >= upper_lim:
        raise ValueError(f"coord must be >= 0 and < {upper_lim} (got {coord})")

    orientation = np.zeros(n, dtype=int)
    for i in range(n - (2 if force_modulo else 1), -1, -1):
        coord, orientation[i] = divmod(coord, v)
    if force_modulo:
        orientation[-1] = -np.sum(orientation[:-1]) % v
    return orientation


def get_orientation_coord(orientation: np.ndarray, v: int, is_modulo: bool = False) -> int:
    """Get orientation coordinate number."""
    if not isinstance(orientation, np.ndarray):
        raise TypeError(f"orientation must be ndarray, not {type(orientation).__name__}")
    if not isinstance(v, int):
        raise TypeError(f"v must be int, not {type(v).__name__}")
    if not isinstance(is_modulo, bool):
        raise TypeError(f"is_modulo must be bool, not {type(is_modulo).__name__}")
    if v <= 0:
        raise ValueError(f"v must be positive (got {v})")
    n = len(orientation)
    if n <= 0:
        raise ValueError(f"orientation length must be positive (got {n})")
    if not np.issubdtype(orientation.dtype, np.integer):
        raise TypeError(f"orientation elements must be int, not {orientation.dtype}")
    if np.any((orientation < 0) | (orientation >= v)):
        raise ValueError(f"orientation values must be >= 0 and < {v} (got {orientation})")
    if is_modulo and np.sum(orientation) % v != 0:
        raise ValueError(f"orientation has no total orientation 0 modulo {v} (got {np.sum(orientation) % v})")

    coord = np.array([0])[0]
    for o in orientation[slice(n - 1 if is_modulo else n)]:
        coord *= v
        coord += o
    return coord.item()


def get_permutation_array(coord: int, n: int, force_even_parity: bool = False) -> Tuple[np.ndarray, bool]:
    """Get permutation array and permutation parity."""
    if not isinstance(coord, int):
        raise TypeError(f"coord must be int, not {type(coord).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, not {type(n).__name__}")
    if not isinstance(force_even_parity, bool):
        raise TypeError(f"force_even_parity must be bool, not {type(force_even_parity).__name__}")
    if not force_even_parity and n <= 0:
        raise ValueError(f"n must be positive (got {n})")
    if force_even_parity and n <= 1:
        raise ValueError(f"n must be > 1 (got {n})")
    try:
        upper_lim = FACTORIAL[n].item()
    except IndexError:
        upper_lim = math.factorial(n)
    if force_even_parity:
        upper_lim //= 2
    if coord < 0 or coord >= upper_lim:
        raise ValueError(f"coord must be >= 0 and < {upper_lim} (got {coord})")

    permutation = np.zeros(n, dtype=int)
    if force_even_parity:
        permutation[-1] = 1
    permutation_parity = 0
    for i in range(n - (3 if force_even_parity else 2), -1, -1):
        coord, permutation[i] = divmod(coord, n - i)
        permutation[i+1:] += permutation[i+1:] >= permutation[i]
        permutation_parity += permutation[i]
    if force_even_parity and bool(permutation_parity % 2):
        permutation[-2:] = permutation[[-1, -2]]
    permutation_parity = False if force_even_parity else bool(permutation_parity % 2)
    return permutation, permutation_parity


def get_permutation_coord(permutation: np.ndarray, is_even_parity: bool = False) -> int:
    """Get permutation coordinate number."""
    if not isinstance(permutation, np.ndarray):
        raise TypeError(f"permutation must be ndarray, not {type(permutation).__name__}")
    if not isinstance(is_even_parity, bool):
        raise TypeError(f"is_even_parity must be bool, not {type(is_even_parity).__name__}")
    n = len(permutation)
    if not is_even_parity and n <= 0:
        raise ValueError(f"permutation length must be positive (got {n})")
    if is_even_parity and n <= 1:
        raise ValueError(f"permutation length must be > 1 (got {n})")
    if not np.issubdtype(permutation.dtype, np.integer):
        raise TypeError(f"permutation elements must be int, not {permutation.dtype}")
    if len(set(permutation)) != n:
        raise ValueError(f"permutation values must be different (got {permutation})")
    if is_even_parity and get_permutation_parity(permutation):
        raise ValueError(f"permutation has no even parity (got {permutation})")

    coord = np.array([0])[0]
    for i in range(n - (2 if is_even_parity else 1)):
        coord *= n - i
        coord += np.sum(permutation[i] > permutation[i+1:])
    return coord.item()


def get_permutation_parity(permutation: np.ndarray) -> bool:
    """Get permutation parity."""
    coord = get_permutation_coord(permutation)
    _, permutation_parity = get_permutation_array(coord, len(permutation))
    return permutation_parity


def get_combination_array(coord: int, n: int) -> np.ndarray:
    """Get combination array."""
    if not isinstance(coord, int):
        raise TypeError(f"coord must be int, not {type(coord).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, not {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"n must be positive (got {n})")
    if coord < 0:
        raise ValueError(f"coord must be >= 0 (got {coord})")

    if n >= COMBINATION.shape[1] or coord >= COMBINATION[-1, n]:
        def comb(n: int, k: int) -> int: return math.comb(n, k)
    else:
        def comb(n: int, k: int) -> int: return COMBINATION[n, k].item()

    i = n - 1
    m = 1
    while coord >= comb(m, n):
        m += 1
    combination = np.zeros(n, dtype=int)
    for c in range(m - 1, 0, -1):
        if coord >= comb(c, i + 1):
            coord -= comb(c, i + 1)
            combination[i] = c
            i -= 1
            if i < 0:
                break
    return combination


def get_combination_coord(combination: np.ndarray) -> int:
    """Get combination coordinate number."""
    if not isinstance(combination, np.ndarray):
        raise TypeError(f"combination must be ndarray, not {type(combination).__name__}")
    n = len(combination)
    if n <= 0:
        raise ValueError(f"combination length must be positive (got {n})")
    if not np.issubdtype(combination.dtype, np.integer):
        raise TypeError(f"combination elements must be int, not {combination.dtype}")
    if np.any(combination < 0):
        raise ValueError(f"combination values must be >= 0 (got {combination})")
    if np.any(np.diff(combination) <= 0):
        raise ValueError(f"combination values must be in increasing order (got {combination})")

    try:
        return np.sum(COMBINATION[combination, range(1, n + 1)]).item()
    except IndexError:
        return np.sum([math.comb(c, i + 1) for i, c in enumerate(combination)]).item()


def get_partial_permutation_array(coord: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Get permutation array and combination array."""
    if not isinstance(coord, int):
        raise TypeError(f"coord must be int, not {type(coord).__name__}")
    if not isinstance(n, int):
        raise TypeError(f"n must be int, not {type(n).__name__}")
    if n <= 0:
        raise ValueError(f"n must be positive (got {n})")
    if coord < 0:
        raise ValueError(f"coord must be >= 0 (got {coord})")

    try:
        comb_coord, perm_coord = divmod(coord, FACTORIAL[n].item())
    except IndexError:
        comb_coord, perm_coord = divmod(coord, math.factorial(n))
    permutation = get_permutation_array(perm_coord, n)[0]
    combination = get_combination_array(comb_coord, n)
    return permutation, combination


def get_partial_permutation_coord(permutation: np.ndarray, combination: np.ndarray) -> int:
    """Ger partial permutation coordinate number."""
    if not isinstance(permutation, np.ndarray):
        raise TypeError(f"permutation must be ndarray, not {type(permutation).__name__}")
    if not isinstance(combination, np.ndarray):
        raise TypeError(f"combination must be ndarray, not {type(combination).__name__}")
    perm_len, comb_len = len(permutation), len(combination)
    if perm_len != comb_len:
        raise ValueError(f"permutation length and combination length must be the same (got {perm_len} != {comb_len})")
    perm_coord = get_permutation_coord(permutation)
    comb_coord = get_combination_coord(combination)
    try:
        return (perm_coord + FACTORIAL[perm_len] * comb_coord).item()
    except IndexError:
        return perm_coord + math.factorial(perm_len) * comb_coord
