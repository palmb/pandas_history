#!/usr/bin/env python

from __future__ import annotations

from functools import reduce

import pandas as pd


class FullSeriesHistory:
    """
    History of a Series:
        - update(series): add new info -> new History
        - restore(index): restore historical status -> Series (from the past)

    optional:
        - concat(other): add another History -> new History
    """

    __change_keys = {'add', 'rm', 'mod'}

    def __init__(self, s: pd.Series | None = None, copy=True):
        self._bases = []
        self._changes = []
        self._current: pd.Series | None = None

    @property
    def empty(self):
        return self._current is None

    def __len__(self):
        return len(self._bases)

    def _get_base_index(self, i=-1):
        base_or_pos = self._bases[i]
        if isinstance(base_or_pos, int):
            return base_or_pos
        return self._to_absolut_index(i)

    def _to_absolut_index(self, i, sz=None):
        """ may raise IndexError for index out of range."""
        return range(len(self) if sz is None else sz)[int(i)]

    def update(self, s: pd.Series, copy=True):
        assert isinstance(s, pd.Series)
        s = s.copy(deep=copy)
        if self.empty:
            self._update(s)
            return self

        rm = self._current.index.difference(s.index)
        if len(rm) == len(self._current):
            self._update(s)
            return self

        add = s.reindex(s.index.difference(self._current.index))
        common = s.index.intersection(self._current.index)
        diff = s.loc[common] != self._current.loc[common]  # todo: or both are nan
        mod = s.loc[common].loc[diff]

        if add.empty:
            add = None
        if rm.empty:
            rm = None
        if mod.empty:
            mod = None

        self._update(dict(add=add, rm=rm, mod=mod))
        return self

    def _update(self, base_or_change):
        """ insert base or change-dict and set '_current'."""
        if isinstance(base_or_change, pd.Series):  # a new base
            return self._update_with_base(base_or_change)
        return self._update_with_change(base_or_change)

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

    def restore(self, i=None):
        if self.empty:
            if i is None:
                return None
            raise IndexError('index out of bounds')
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
        changes = [d for d in self._changes[base_i + 1:i + 1] if d is not None]
        for change in changes:
            assert isinstance(change, dict)
            result = self._apply(result, change)
        return result

    @staticmethod
    def _apply(base: pd.Series, change):
        add: pd.Series = change['add']
        rm: pd.Index = change['rm']
        mod: pd.Series = change['mod']
        if add is not None:
            base = base.reindex(base.index.union(add.index))
            base.loc[add.index] = add
        if mod is not None:
            base.loc[mod.index] = mod
        if rm is not None:
            base = base.drop(rm)
        return base

    def __repr__(self):
        if self.empty:
            return f"empty {self.__class__.__name__}"
        return str(self._current)


def test_empty():
    h = FullSeriesHistory()
    assert h.empty
    h.update(pd.Series([], dtype=float))
    assert not h.empty


def get_size(obj):
    import pickle
    return len(pickle.dumps(obj))


if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore', ".*iteritems")
    warnings.filterwarnings('ignore', ".*with an integer-dtype index is deprecated")
    test_empty()
    h = FullSeriesHistory()
    s1 = pd.Series([1, 2])
    s2 = pd.Series([1, 2, 3])
    s3 = pd.Series(2, index=[10, 20, 30])
    h.update(s1)
    h.update(s2)
    h.update(s1)
    h.update(s3)
    h.update(s1)
    print(h.restore())
