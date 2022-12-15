#!/usr/bin/env python
import warnings

from core import SeriesHistory
import pandas as pd
import numpy as np

# todo:
#  - optimize for dtype in __new__ ?
#  - optimize for speed -> dict ??
#  - general optimize: sparse with value-count

# todo Flags:
#   - wrapper to fit old saqc-Flags
#   - squeeze (since last index change)


class FixedDtypeSeriesHistory(SeriesHistory):

    def __init__(self, s: pd.Series | None = None, copy=True, dtype=None):
        if dtype is None:
            raise ValueError("A dtype is needed")
        self._dtype = pd.Series(dtype=dtype).dtype
        super().__init__(s, copy)

    def _prepare_input(self, obj, copy=True, index_change=False, **kwargs) -> pd.Series:
        obj = super()._prepare_input(obj, copy, **kwargs)
        if not pd.api.types.is_dtype_equal(obj.dtype, self._dtype):
            raise ValueError(f"series dtype must be {self._dtype}, not {obj.dtype}.")
        return obj

    def _apply(self, base: pd.Series, change: dict | None) -> pd.Series:
        if change is not None:
            change.pop('dtype', None)
        base = super()._apply(base, change)
        try:
            base = base.astype(self._dtype, copy=False)
        except (TypeError, ValueError):
            warnings.warn(f"could not cast to requested dtype {self._dtype}")
        return base


class ValueHistory(SeriesHistory):

    def _prepare_input(self, obj, copy=True, index_change=False, **kwargs):
        super()._prepare_input(obj, copy, **kwargs)
        assert pd.api.types.is_float_dtype(obj.dtype)
        return obj

    def update(self, series: pd.Series, copy=True, index_change=False, **kwargs):
        return super().update(series, copy, index_change=index_change, **kwargs)
