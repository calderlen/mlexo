__all__ = ["ObjectStack"]

from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Any, Generic, TypeVar

import mlx.core as mx

Obj = TypeVar("Obj")


def _is_container(value: Any) -> bool:
    return isinstance(value, (dict, list, tuple))


def _slice_leaf(value: Any, axis: int | None, index: int) -> Any:
    if axis is None:
        return value
    idx = [slice(None)] * axis + [index]
    return value[tuple(idx)]


def _slice_tree(value: Any, axes: Any, index: int) -> Any:
    if axes is None:
        return value
    if isinstance(value, dict):
        if isinstance(axes, Mapping):
            return {key: _slice_tree(value[key], axes[key], index) for key in value}
        return {key: _slice_tree(child, axes, index) for key, child in value.items()}
    if isinstance(value, tuple):
        if isinstance(axes, tuple):
            return tuple(
                _slice_tree(child, axis, index)
                for child, axis in zip(value, axes, strict=True)
            )
        return tuple(_slice_tree(child, axes, index) for child in value)
    if isinstance(value, list):
        if isinstance(axes, list):
            return [
                _slice_tree(child, axis, index)
                for child, axis in zip(value, axes, strict=True)
            ]
        return [_slice_tree(child, axes, index) for child in value]
    return _slice_leaf(value, axes, index)


def _stack_tree(parts: list[Any], axes: Any) -> Any:
    if not parts:
        raise ValueError("Cannot stack an empty result tree")
    if axes is None:
        return parts[0]

    first = parts[0]
    if isinstance(first, dict):
        child_axes = axes if isinstance(axes, Mapping) else {key: axes for key in first}
        return {
            key: _stack_tree([part[key] for part in parts], child_axes[key])
            for key in first
        }
    if isinstance(first, tuple):
        child_axes = axes if isinstance(axes, tuple) else tuple(axes for _ in first)
        return tuple(
            _stack_tree([part[i] for part in parts], child_axes[i])
            for i in range(len(first))
        )
    if isinstance(first, list):
        child_axes = axes if isinstance(axes, list) else [axes for _ in first]
        return [
            _stack_tree([part[i] for part in parts], child_axes[i])
            for i in range(len(first))
        ]
    return mx.stack(parts, axis=axes)


def _normalize_in_axes(args: tuple[Any, ...], in_axes: Any) -> tuple[Any, ...]:
    if not args:
        return ()
    if isinstance(in_axes, tuple):
        return in_axes
    if isinstance(in_axes, list):
        return tuple(in_axes)
    if isinstance(in_axes, Mapping):
        if len(args) != 1:
            raise ValueError("Structured in_axes only supports a single positional arg")
        return (in_axes,)
    return tuple(in_axes for _ in args)


class ObjectStack(Generic[Obj]):
    """A stack of objects supporting a JAX-like vmap interface.

    The MLX port keeps the public API but currently always evaluates using a Python loop
    over the objects. This avoids depending on JAX's pytree internals while keeping the
    behavior of ``System.body_vmap`` and related helpers.
    """

    objects: tuple[Obj, ...]
    stack: Obj | None

    def __init__(self, *objects: Obj):
        self.objects = objects
        self.stack = None

    def __len__(self) -> int:
        return len(self.objects)

    def vmap(
        self,
        func: Callable,
        in_axes: int | None | Sequence[Any] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        @wraps(func)
        def impl(*args):
            if not self.objects:
                raise ValueError("Cannot map over an empty ObjectStack")

            in_axes_ = _normalize_in_axes(args, in_axes)
            results = []
            for n, obj in enumerate(self.objects):
                indexed_args = tuple(
                    _slice_tree(arg, axis, n)
                    for arg, axis in zip(args, in_axes_, strict=True)
                )
                results.append(func(obj, *indexed_args))
            return _stack_tree(results, out_axes)

        return impl
