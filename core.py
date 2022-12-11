#!/usr/bin/env python

from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd


class FullSeriesHistory:
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
        if add+mod > BASE_FAKTOR * last series,
        we store the entire new series
    """

    BASE_FAKTOR = 0.67
    __change_keys = {"add", "rm", "mod"}

    def __init__(self, s: pd.Series | None = None, copy=True):
        self._bases = []
        self._changes = []
        self._current: pd.Series | None = None
        if s is None:
            s = pd.Series(dtype=float)
            copy = False
        if copy:
            s = s.copy()
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

    def update(self, series: pd.Series, skipna=False,  copy=True):
        """
        skipna: bool
            Only update valid values. In other words,
            keep the previous value if the given series is NaN.
        """
        assert isinstance(series, pd.Series)
        s = series.copy(deep=copy)

        # differences in index
        add = s.reindex(s.index.difference(self._current.index))
        rm = self._current.index.difference(s.index)
        # differences in values
        mod = self._get_modified(s, skipna)

        # optimisation
        if (
            (not skipna or not s.hasnans)
            and len(mod) + len(add) > self.BASE_FAKTOR * len(self._current)
        ):
            return self._set_new_base(s)
        if add.empty:
            add = None
        if rm.empty:
            rm = None
        if mod.empty:
            mod = None
        if add is None and rm is None and mod is None:
            return self._update(None)
        
        return self._update(dict(add=add, rm=rm, mod=mod))

    def _get_modified(self, s, skipna):
        if skipna or not s.hasnans:
            common = s.dropna().index.intersection(self._current.index)
            s, c = s.reindex(common), self._current.reindex(common)
            return s.loc[~(s == c)]

        # 1. calculate rows where either the new
        #    series xor the current is NaN.
        # 2. calculate rows where the non-NaN values differ
        common = s.index.intersection(self._current.index)
        s, c = s.reindex(common), self._current.reindex(common)
        return s.loc[(s.isna() ^ c.isna()) | ~(s == c)]

    def _update(self, base_or_change):
        """insert base or change-dict and set '_current'."""
        if isinstance(base_or_change, pd.Series):  # a new base
            return self._update_with_base(base_or_change)
        return self._update_with_change(base_or_change)

    _set_new_base = _update  # alias

    def _update_with_base(self, base):
        self._bases.append(base)
        self._changes.append(None)
        self._current = self._bases[-1]

    def _update_with_change(self, change):
        if change is None:
            pass
        elif isinstance(change, dict):
            assert self.__change_keys == change.keys()
            if all(map(lambda x: x is None, change.values())):
                change = None
        else:
            raise TypeError(type(change))
        if change is not None:
            # first call self._apply to ensure the data
            # is valid, before storing it
            self._current = self._apply(self._current.copy(), change)
        self._changes.append(change)
        self._bases.append(self._get_base_index())

    def get(self):
        return self.restore(i=None)

    def restore(self, i=None):
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
            assert isinstance(change, dict)
            result = self._apply(result, change)
        return result

    @staticmethod
    def _apply(base: pd.Series, change):
        add: pd.Series = change["add"]
        rm: pd.Index = change["rm"]
        mod: pd.Series = change["mod"]
        if add is not None:
            base = base.reindex(base.index.union(add.index))
            base.loc[add.index] = add
        if mod is not None:
            base.loc[mod.index] = mod
        if rm is not None:
            base = base.drop(rm)
        return base

    def pprint(self):
        serieses = []
        for i in range(len(self)):
            serieses.append(self.restore(i))
        index = reduce(pd.Index.union, [s.index for s in serieses], pd.Index([]))
        df = pd.DataFrame(dict(zip(range(len(serieses)), serieses)), index=index)
        df = df.astype(str)
        for i in df.columns:
            idx = df.index.difference(serieses[i].index)
            df.loc[idx, i] = ''

        bases = self._bases
        columns = [i if isinstance(bases[i], int) else f'{i}(base)' for i in df.columns]
        df.columns = columns

        print(df)

    def __repr__(self):
        return str(self._current)


def test_empty():
    h = FullSeriesHistory()
    assert h.empty
    h.update(pd.Series([], dtype=float))
    assert not h.empty


def get_size(obj):
    import pickle

    return len(pickle.dumps(obj))


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", ".*iteritems")
    warnings.filterwarnings("ignore", ".*with an integer-dtype index is deprecated")
    i0 = pd.date_range('2000', None, 20)
    s0 = pd.Series(np.nan, index=i0, dtype=float)
    h = FullSeriesHistory(s0)
    print(get_size(h))

    # flag stuff
    curr = h.get()
    curr.iloc[::2] = 10
    h.update(curr, skipna=True)
    print(get_size(h))

    # reindex stuff
    i1 = pd.date_range('2000-01-10', None, 15)
    curr = h.get().reindex(i1)
    h.update(curr, skipna=True)
    print(get_size(h))

    # flag and reindex stuff
    curr = h.get().shift(freq='-1d')
    curr.iloc[7:10] = 25.3
    h.update(curr, skipna=False)
    print(get_size(h))

    # flag nothing
    curr = h.get()
    curr.iloc[:] = np.nan
    h.update(curr, skipna=True)
    print(get_size(h))

    # flag all
    curr = h.get()
    curr.iloc[:] = 99
    h.update(curr, skipna=True)
    print(get_size(h))

    h.pprint()
