#!/usr/bin/env python

from __future__ import annotations

import logging
from functools import reduce
from typing import final

import numpy as np
import pandas as pd


class SeriesHistory:
    """
    History of a Series:
        - update(series): add new info -> new History
        - restore(index): restore historical status -> Series (from the past)

    optional:
        - concat(other): add another History -> new History

    internal:
        partition into:
            - index to remove
            - index+value to add
            - index+value to modify
        if length of add+mod > BASE_FACTOR * length of last series - rm,
        we store the entire new series
    """

    BASE_FACTOR = 0.6
    __change_keys = {"add", "rm", "mod", "dtype"}

    def __init__(self, s: pd.Series | None = None, copy=True):
        # _bases: stores the base or the index of the last base in the list
        # _changes: stores a dict of changes or None for no changes
        # _current: is a reference or a copy of current value of the series
        self._bases: list[pd.Series | int] = []
        self._changes: list[dict[str, pd.Series | pd.Index | None] | None] = []
        self._current: pd.Series | None = None
        if s is None:
            s, copy = pd.Series(dtype=float), False
        s = self._prepare_input(s, copy=copy)
        self._set_new_base(s)

    def __len__(self):
        return len(self._bases)

    def _get_base_index(self, i=-1):
        base_or_pos = self._bases[i]
        if isinstance(base_or_pos, int):
            return base_or_pos
        return self._to_absolut_index(i)

    def _to_absolut_index(self, i, sz=None):
        """may raise IndexError for index out of range."""
        return range(len(self) if sz is None else sz)[int(i)]

    def _prepare_input(self, obj, copy=True, **kwargs) -> pd.Series:
        assert isinstance(obj, pd.Series)
        logging.debug(f"len: {len(obj)}")
        if copy:
            obj = obj.copy()
        return obj

    def update(self, series: pd.Series, copy=True, **kwargs):
        s = self._prepare_input(series, copy=copy, **kwargs)
        rm = self._get_rm(s, **kwargs)
        add = self._get_add(s, **kwargs)
        mod = self._get_modified(s, **kwargs)
        dtype = self._get_dtype(s, **kwargs)

        logging.debug(
            f"rm: {len(rm)}, add: {len(add)}, mod: {len(mod)} => {len(mod) + len(add)} >? {self.BASE_FACTOR * (len(self._current) - len(rm))}"
        )

        if len(mod) + len(add) > self.BASE_FACTOR * (len(self._current) - len(rm)):
            return self._set_new_base(s)

        to_update = dict()
        if not rm.empty:
            to_update["rm"] = rm
        if not add.empty:
            to_update["add"] = add
        if not mod.empty:
            to_update["mod"] = mod
        if dtype != self._current.dtype:
            to_update["dtype"] = dtype
        to_update = to_update or None

        try:
            assert s.equals(self._apply(self._current.copy(), to_update))
        except AssertionError as e:
            logging.debug(f"not equal", exc_info=e)
            to_update = s
        except (TypeError, IndexError, ValueError, KeyError) as e:
            logging.debug(f"_apply failed", exc_info=e)
            to_update = s
        finally:
            self._update(to_update)
        return self

    def _get_rm(self, s, **kwargs) -> pd.Index:
        return self._current.index.difference(s.index)

    def _get_add(self, s, **kwargs) -> pd.Series:
        return s.reindex(s.index.difference(self._current.index))

    def _get_dtype(self, s, **kwargs):
        return s.dtype

    def _get_modified(self, s, **kwargs) -> pd.Series:
        common = s.index.intersection(self._current.index)
        s, c = s.reindex(common), self._current.reindex(common)
        both_nan = c.isna() & s.isna()
        value_change = (c != s) & ~both_nan
        identical = (c == s) | both_nan
        logging.debug(f"identical: {len(identical[identical])}")
        return s.loc[value_change]

    def _update(self, base_or_change) -> None:
        """insert base or change-dict and set '_current'."""
        if isinstance(base_or_change, pd.Series):  # a new base
            self._bases.append(base_or_change)
            self._changes.append(None)
            self._current = self._bases[-1]
        else:
            if base_or_change is not None:
                assert isinstance(base_or_change, dict)
                self._current = self._apply(self._current.copy(), base_or_change)
            self._changes.append(base_or_change)
            self._bases.append(self._get_base_index())

    _set_new_base = _update  # alias

    def get(self) -> pd.Series:
        return self.restore(i=None)

    def restore(self, i=None) -> pd.Series:
        if i is None:
            return self._current.copy()
        try:
            i = self._to_absolut_index(i)
        except Exception as e:
            raise type(e)(*e.args) from None
        base_i = self._get_base_index(i)
        base = self._bases[base_i]
        assert isinstance(base, pd.Series)
        result = base.copy()
        changes = [d for d in self._changes[base_i + 1 : i + 1] if d is not None]
        for change in changes:
            result = self._apply(result, change)
        return result

    def _apply(self, base: pd.Series, change: dict | None) -> pd.Series:
        """ may change base, copy explicitly before if needed. """
        if change is None:
            return base
        rm: pd.Index = change.get("rm")
        add: pd.Series = change.get("add")
        mod: pd.Series = change.get("mod")
        dtype = change.get("dtype")
        index = self._construct_index(base, add, rm)
        base = base.reindex(index)
        if add is not None:
            base.loc[add.index] = add
        if mod is not None:
            base.loc[mod.index] = mod
        if dtype is not None:
            base = base.astype(dtype)
        return base

    def _construct_index(self, s: pd.Series, add: pd.Series, rm: pd.Index) -> pd.Index:
        index = s.index
        if rm is not None:
            index = index.difference(rm)
        if add is not None:
            index = index.union(add.index)
        return index.sort_values()

    def pprint(self):
        serieses = []
        for i in range(len(self)):
            serieses.append(self.restore(i))
        index = reduce(pd.Index.union, [s.index for s in serieses], pd.Index([]))
        df = pd.DataFrame(dict(zip(range(len(serieses)), serieses)), index=index)
        df = df.astype(str)
        for i in df.columns:
            idx = df.index.difference(serieses[i].index)
            df.loc[idx, i] = ""

        bases = self._bases
        columns = [i if isinstance(bases[i], int) else f"{i}(base)" for i in df.columns]
        df.columns = columns
        print(df)

    def __repr__(self):
        return str(self._current)


def get_size(obj):
    import pickle

    return len(pickle.dumps(obj))


def foo():
    i0 = pd.date_range("2000", None, 20)
    s0 = pd.Series(np.nan, index=i0, dtype=float)
    h = SeriesHistory(s0)
    print(get_size(h))

    # flag all stuff, change dtype
    s = s0.copy()
    s.iloc[:] = "foo"
    h.update(s)
    print(get_size(h))

    # flag stuff
    curr = h.get()
    curr.iloc[::2] = 10
    h.update(curr)
    print(get_size(h))

    # flag same stuff
    s = s0.copy()
    s.iloc[5::2] = 10
    h.update(s)
    print(get_size(h))

    # reindex stuff
    i1 = pd.date_range("2000-01-10", None, 15)
    curr = h.get().reindex(i1)
    h.update(curr)
    print(get_size(h))

    # flag and reindex stuff
    curr = h.get().shift(freq="-1d")
    curr.iloc[7:10] = 25.3
    h.update(curr)
    print(get_size(h))

    # flag nothing
    curr = h.get()
    curr.iloc[:] = np.nan
    h.update(curr)
    print(get_size(h))

    # flag all
    curr = h.get()
    curr.iloc[:] = 99
    h.update(curr)
    print(get_size(h))

    h.pprint()


def identical_test():
    i0 = pd.date_range("2000", None, 20)
    s0 = pd.Series(np.nan, index=i0, dtype=float)
    h = SeriesHistory(s0)
    s1 = s0.copy()
    s1.iloc[:4] = 99
    h.update(s1)
    h.update(s1)
    s2 = s1.copy()
    s2 = s2.iloc[:4]
    h.update(s2)
    h.pprint()
    assert h.restore(0).equals(s0)
    assert h.restore(1).equals(s1)
    assert h.restore(2).equals(s1)
    assert h.restore(3).equals(s2)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", ".*iteritems")
    warnings.filterwarnings("ignore", ".*with an integer-dtype index is deprecated")
    logging.basicConfig(level="DEBUG")
    # identical_test()
    foo()
