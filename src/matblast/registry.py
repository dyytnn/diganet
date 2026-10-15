from __future__ import annotations
from typing import Callable, TypeVar

T = TypeVar("T")


class Registry(dict[str, T]):
    """Small explicit registry mirroring NST_segmentation's named builders."""
    def register(self, name: str):
        def decorator(obj: T) -> T:
            if name in self:
                raise KeyError(f"{name!r} already registered")
            self[name] = obj
            return obj
        return decorator

    def build(self, name: str, **kwargs):
        if name not in self:
            raise KeyError(f"Unknown registry entry {name!r}; available: {sorted(self)}")
        return self[name](**kwargs)
