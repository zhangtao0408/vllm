# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class KVCacheTensorLike(Protocol):
    size: int
    shared_by: list[str]


@dataclass(frozen=True)
class BucketSlot:
    page_size: int
    slot_idx: int
    layer_names: tuple[str, ...]


@dataclass(frozen=True)
class ShadowSource:
    page_size: int
    slot_idx: int
    layer_name: str


@dataclass(frozen=True)
class ShadowPlacement:
    target_page_size: int
    target_slot_idx: int
    target_layer_names: tuple[str, ...]
    source: ShadowSource
    source_bucket_keys: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class MissingSourceCacheError(Exception):
    target_layer_name: str

    def __str__(self) -> str:
        return f"No producer cache matches target cache {self.target_layer_name!r}."


@dataclass(frozen=True)
class DuplicateSourceCacheError(Exception):
    source_layer_name: str

    def __str__(self) -> str:
        return (
            f"Producer cache {self.source_layer_name!r} appears in multiple "
            "bucket slots."
        )


@dataclass(frozen=True)
class ShadowBucketConfigError(Exception):
    reason: str

    def __str__(self) -> str:
        return f"Invalid Mooncake shadow bucket config: {self.reason}"


def canonical_cache_name(layer_name: str) -> str:
    return layer_name.replace(".self_attn.attn", ".attn").replace(
        ".self_attn.", ".attn."
    )


def bucket_slots_from_kv_cache_tensors(
    kv_cache_tensors: Sequence[KVCacheTensorLike],
    num_blocks: int,
) -> tuple[BucketSlot, ...]:
    slot_counts: dict[int, int] = defaultdict(int)
    buckets: list[BucketSlot] = []
    for kv_cache_tensor in kv_cache_tensors:
        page_size = kv_cache_tensor.size // num_blocks
        slot_idx = slot_counts[page_size]
        slot_counts[page_size] += 1
        buckets.append(
            BucketSlot(
                page_size=page_size,
                slot_idx=slot_idx,
                layer_names=tuple(kv_cache_tensor.shared_by),
            )
        )
    return tuple(buckets)


def parse_shadow_buckets(raw_buckets: Any) -> tuple[BucketSlot, ...]:
    if isinstance(raw_buckets, str):
        raw_buckets = _load_shadow_bucket_file(raw_buckets)
    if not isinstance(raw_buckets, Sequence) or isinstance(raw_buckets, str):
        raise ShadowBucketConfigError("shadow_buckets must be a list of bucket objects")

    slot_counts: dict[int, int] = defaultdict(int)
    buckets: list[BucketSlot] = []
    for raw_bucket in raw_buckets:
        if not isinstance(raw_bucket, Mapping):
            raise ShadowBucketConfigError("each shadow bucket must be an object")
        page_size = _parse_int(raw_bucket, "page_size")
        layer_names = _parse_layer_names(raw_bucket)
        raw_slot_idx = raw_bucket.get("slot_idx")
        if raw_slot_idx is None:
            slot_idx = slot_counts[page_size]
        elif isinstance(raw_slot_idx, int):
            slot_idx = raw_slot_idx
        else:
            raise ShadowBucketConfigError("slot_idx must be an integer when provided")
        slot_counts[page_size] = max(slot_counts[page_size], slot_idx + 1)
        buckets.append(
            BucketSlot(
                page_size=page_size,
                slot_idx=slot_idx,
                layer_names=layer_names,
            )
        )
    return tuple(buckets)


def _load_shadow_bucket_file(config_path: str) -> Any:
    path = Path(config_path).expanduser()
    try:
        with path.open(encoding="utf-8") as config_file:
            raw_config = json.load(config_file)
    except OSError as exc:
        raise ShadowBucketConfigError(
            f"failed to read shadow_buckets file {config_path!r}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ShadowBucketConfigError(
            f"failed to parse shadow_buckets file {config_path!r}: {exc}"
        ) from exc

    if isinstance(raw_config, Mapping) and "shadow_buckets" in raw_config:
        return raw_config["shadow_buckets"]
    return raw_config


def build_shadow_bucket_plan(
    producer_buckets: Sequence[BucketSlot],
    consumer_buckets: Sequence[BucketSlot],
) -> tuple[ShadowPlacement, ...]:
    producer_slots = _index_producer_buckets(producer_buckets)
    placements: list[ShadowPlacement] = []

    for target_bucket in consumer_buckets:
        source_candidates: dict[tuple[int, int], ShadowSource] = {}
        for target_layer_name in target_bucket.layer_names:
            source = producer_slots.get(canonical_cache_name(target_layer_name))
            if source is None:
                raise MissingSourceCacheError(target_layer_name)
            source_candidates[(source.page_size, source.slot_idx)] = source

        source = next(iter(source_candidates.values()))
        placements.append(
            ShadowPlacement(
                target_page_size=target_bucket.page_size,
                target_slot_idx=target_bucket.slot_idx,
                target_layer_names=target_bucket.layer_names,
                source=source,
                source_bucket_keys=tuple(source_candidates),
            )
        )

    return tuple(placements)


def _parse_int(raw_bucket: Mapping[Any, Any], key: str) -> int:
    value = raw_bucket.get(key)
    if not isinstance(value, int):
        raise ShadowBucketConfigError(f"{key} must be an integer")
    return value


def _parse_layer_names(raw_bucket: Mapping[Any, Any]) -> tuple[str, ...]:
    raw_layer_names = raw_bucket.get("layer_names")
    if not isinstance(raw_layer_names, Sequence) or isinstance(raw_layer_names, str):
        raise ShadowBucketConfigError("layer_names must be a list of strings")
    layer_names: list[str] = []
    for layer_name in raw_layer_names:
        if not isinstance(layer_name, str):
            raise ShadowBucketConfigError("layer_names must contain only strings")
        layer_names.append(layer_name)
    if not layer_names:
        raise ShadowBucketConfigError("layer_names must not be empty")
    return tuple(layer_names)


def _index_producer_buckets(
    producer_buckets: Sequence[BucketSlot],
) -> dict[str, ShadowSource]:
    slots: dict[str, ShadowSource] = {}
    for bucket in producer_buckets:
        for layer_name in bucket.layer_names:
            canonical_name = canonical_cache_name(layer_name)
            if canonical_name in slots:
                raise DuplicateSourceCacheError(layer_name)
            slots[canonical_name] = ShadowSource(
                page_size=bucket.page_size,
                slot_idx=bucket.slot_idx,
                layer_name=layer_name,
            )
    return slots
