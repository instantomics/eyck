import os
import pathlib
import pickle
import pandas as pd

from .files import get_git_root, get_output, get_code, get_data, get_software, get_results
from .h5ad import write_h5ad

__all__ = [
    "get_git_root",
    "get_output",
    "get_code",
    "get_data",
    "get_software",
    "get_results",
    "write_h5ad",
]

def fix_class(obj):
    import importlib

    module = importlib.import_module(obj.__class__.__module__)
    cls = getattr(module, obj.__class__.__name__)
    try:
        obj.__class__ = cls
    except TypeError:
        pass


class Pickler(pickle.Pickler):
    def reducer_override(self, obj):
        if any(obj.__class__.__module__.startswith(module) for module in ["chromatinhd."]):
            fix_class(obj)
        else:
            # For any other object, fallback to usual reduction
            return NotImplemented

        return NotImplemented


def save(obj, fh, pickler=None, **kwargs):
    if pickler is None:
        pickler = Pickler
    return pickler(fh).dump(obj)


def crossing(*dfs, **kwargs):
    dfs = [df.copy() if isinstance(df, pd.DataFrame) else df.to_frame() for df in dfs]
    dfs.extend(pd.DataFrame({k: v}) for k, v in kwargs.items())
    for df in dfs:
        df["___key"] = 0
    if len(dfs) == 0:
        return pd.DataFrame()
    dfs = [df for df in dfs if df.shape[0] > 0]  # remove empty dfs
    base = dfs[0]
    for df in dfs[1:]:
        base = pd.merge(base, df, on="___key")
    return base.drop(columns=["___key"])


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("peakfreeatac"):
            module = module.replace("peakfreeatac", "chromatinhd")
        return super().find_class(module, name)


def load(file):
    return Unpickler(file).load()


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)
