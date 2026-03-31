from typing import Any

import mlx.core as mx


def asarray(value: Any, dtype: mx.Dtype | None = None):
    if value is None:
        return None
    if dtype is None:
        return mx.array(value)
    return mx.array(value, dtype=dtype)


def ndim(value: Any) -> int:
    return int(mx.array(value).ndim)
