# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from .xfer_debug_descriptor import (
    XferDebugDescriptor,
    descriptor_from_dict,
    descriptor_to_dict,
)

PointerKind = Literal["source", "destination"]

__all__ = (
    "PointerKind",
    "XferDebugCacheView",
    "XferDebugConfig",
    "XferDebugDescriptor",
    "XferDebugDumpRequest",
    "XferDebugReadError",
    "XferDebugRegionInfo",
    "descriptor_from_dict",
    "descriptor_to_dict",
    "dump_xfer_debug_records",
    "parse_xfer_debug_config",
    "read_debug_bytes",
)


@dataclass(frozen=True)
class XferDebugConfig:
    dump_dir: Path
    max_requests: int = 1
    dump_full_source_block: bool = False
    dump_on_prefill: bool = False


@dataclass(frozen=True)
class XferDebugCacheView:
    layer_name: str
    tensor: torch.Tensor
    base_addr: int
    block_len: int
    materialized_block_len: int | None = None

    @property
    def end_addr(self) -> int:
        return self.base_addr + self.tensor.shape[0] * self.block_len

    @property
    def readable_block_len(self) -> int:
        if self.materialized_block_len is None:
            return self.block_len
        return self.materialized_block_len


@dataclass(frozen=True)
class XferDebugRegionInfo:
    source_layer: str
    target_layers: tuple[str, ...]
    source_block_len: int
    target_block_len: int

    @property
    def bucket_type(self) -> str:
        if self.target_block_len == 1728:
            return "c128_1728"
        return str(self.target_block_len)


@dataclass(frozen=True)
class XferDebugDumpRequest:
    config: XferDebugConfig
    cache_views: tuple[XferDebugCacheView, ...]
    side: str
    pointer_kind: PointerKind
    descriptors: tuple["XferDebugDescriptor", ...]
    rank_tag: str | None = None


@dataclass(frozen=True)
class XferDebugReadError(RuntimeError):
    ptr: int
    length: int
    reason: str

    def __str__(self) -> str:
        return (
            "Failed to read Mooncake xfer debug bytes at "
            f"ptr={self.ptr} length={self.length}: {self.reason}"
        )


def parse_xfer_debug_config(raw_config: Any) -> XferDebugConfig | None:
    if raw_config is None:
        return None
    if isinstance(raw_config, str):
        raw_config = _load_xfer_debug_config(raw_config)
    if not isinstance(raw_config, Mapping):
        msg = "kv_connector_extra_config.xfer_debug must be an object"
        raise TypeError(msg)

    raw_dump_dir = raw_config.get("dump_dir")
    if not isinstance(raw_dump_dir, str) or not raw_dump_dir:
        msg = "xfer_debug.dump_dir must be a non-empty string"
        raise TypeError(msg)

    raw_max_requests = raw_config.get("max_requests", 1)
    if not isinstance(raw_max_requests, int):
        msg = "xfer_debug.max_requests must be an integer"
        raise TypeError(msg)

    return XferDebugConfig(
        dump_dir=Path(raw_dump_dir).expanduser(),
        max_requests=max(raw_max_requests, 0),
        dump_full_source_block=bool(raw_config.get("dump_full_source_block", False)),
        dump_on_prefill=bool(raw_config.get("dump_on_prefill", False)),
    )


def _load_xfer_debug_config(raw_config: str) -> Any:
    stripped_config = raw_config.strip()
    if stripped_config.startswith("{"):
        return json.loads(stripped_config)
    return json.loads(Path(stripped_config).expanduser().read_text(encoding="utf-8"))


def dump_xfer_debug_records(request: XferDebugDumpRequest) -> int:
    request.config.dump_dir.mkdir(parents=True, exist_ok=True)
    dumped = 0
    for descriptor in request.descriptors:
        ptr = (
            descriptor.src_ptr
            if request.pointer_kind == "source"
            else descriptor.dst_ptr
        )
        preferred_names = _preferred_tensor_names(request.pointer_kind, descriptor)
        payload_view = _find_cache_view(request.cache_views, ptr, preferred_names)
        payload_padding_num_bytes = _padding_bytes_in_range(
            request.cache_views,
            ptr,
            descriptor.length,
            preferred_names,
        )
        payload = read_debug_bytes(
            request.cache_views,
            ptr,
            descriptor.length,
            preferred_names,
        )
        record = _build_record(
            side=request.side,
            pointer_kind=request.pointer_kind,
            descriptor=descriptor,
            payload=payload,
            tensor_dtype=str(payload_view.tensor.dtype),
            tensor_shape=tuple(payload_view.tensor.shape),
            tensor_name=payload_view.layer_name,
            block_stride_bytes=payload_view.block_len,
            materialized_block_bytes=payload_view.readable_block_len,
            payload_padding_num_bytes=payload_padding_num_bytes,
            rank_tag=request.rank_tag,
            full_source_block=_read_full_source_block(request, descriptor),
        )
        torch.save(
            record,
            _record_path(
                request.config.dump_dir,
                request.side,
                descriptor,
                request.rank_tag,
                payload_view.layer_name,
            ),
        )
        dumped += 1
    return dumped


def read_debug_bytes(
    cache_views: Sequence[XferDebugCacheView],
    ptr: int,
    length: int,
    preferred_layer_names: tuple[str, ...] = (),
) -> torch.Tensor:
    chunks: list[bytes] = []
    remaining = length
    cursor = ptr

    while remaining > 0:
        view = _find_cache_view(cache_views, cursor, preferred_layer_names)
        block_id = (cursor - view.base_addr) // view.block_len
        block_offset = (cursor - view.base_addr) % view.block_len
        chunk_len = min(remaining, view.block_len - block_offset)
        block_bytes = _block_stride_bytes(view, block_id)
        end_offset = block_offset + chunk_len
        if end_offset > len(block_bytes):
            raise XferDebugReadError(
                ptr=cursor,
                length=chunk_len,
                reason=(
                    f"requested byte range [{block_offset}, {end_offset}) "
                    f"exceeds available block bytes {len(block_bytes)} "
                    f"for layer {view.layer_name!r}"
                ),
            )
        chunks.append(block_bytes[block_offset:end_offset])
        cursor += chunk_len
        remaining -= chunk_len

    return _uint8_tensor_from_bytes(b"".join(chunks))


def _padding_bytes_in_range(
    cache_views: Sequence[XferDebugCacheView],
    ptr: int,
    length: int,
    preferred_layer_names: tuple[str, ...] = (),
) -> int:
    padding_bytes = 0
    remaining = length
    cursor = ptr
    while remaining > 0:
        view = _find_cache_view(cache_views, cursor, preferred_layer_names)
        block_offset = (cursor - view.base_addr) % view.block_len
        chunk_len = min(remaining, view.block_len - block_offset)
        end_offset = block_offset + chunk_len
        materialized_end = min(end_offset, view.readable_block_len)
        padding_bytes += max(end_offset - max(block_offset, materialized_end), 0)
        cursor += chunk_len
        remaining -= chunk_len
    return padding_bytes


def _find_cache_view(
    cache_views: Sequence[XferDebugCacheView],
    ptr: int,
    preferred_layer_names: tuple[str, ...] = (),
) -> XferDebugCacheView:
    if preferred_layer_names:
        preferred_layer_names_set = set(preferred_layer_names)
        for view in cache_views:
            if (
                view.layer_name in preferred_layer_names_set
                and view.base_addr <= ptr < view.end_addr
            ):
                return view
    for view in cache_views:
        if view.base_addr <= ptr < view.end_addr:
            return view
    raise XferDebugReadError(
        ptr=ptr,
        length=0,
        reason="pointer is outside registered KV cache debug views",
    )


def _block_stride_bytes(view: XferDebugCacheView, block_id: int) -> bytes:
    if block_id < 0 or block_id >= view.tensor.shape[0]:
        raise XferDebugReadError(
            ptr=view.base_addr + block_id * view.block_len,
            length=view.block_len,
            reason=(
                f"block_id={block_id} is outside tensor with "
                f"{view.tensor.shape[0]} blocks"
            ),
        )
    raw_storage_bytes = _raw_storage_block_bytes(view, block_id)
    if raw_storage_bytes is not None:
        return raw_storage_bytes

    block = view.tensor[block_id].detach().contiguous().cpu()
    try:
        block_bytes = block.numpy().tobytes()
    except (TypeError, RuntimeError):
        block_bytes = bytes(block.view(torch.uint8).tolist())

    if len(block_bytes) >= view.block_len:
        return block_bytes[: view.block_len]
    return block_bytes + bytes(view.block_len - len(block_bytes))


def _raw_storage_block_bytes(
    view: XferDebugCacheView,
    block_id: int,
) -> bytes | None:
    try:
        storage = view.tensor.untyped_storage()
        storage_offset = view.tensor.storage_offset() * view.tensor.element_size()
        byte_offset = storage_offset + block_id * view.block_len
        if byte_offset < 0 or byte_offset + view.block_len > storage.nbytes():
            return None
        byte_tensor = torch.empty(
            0,
            dtype=torch.uint8,
            device=view.tensor.device,
        ).set_(storage, byte_offset, (view.block_len,), (1,))
        byte_tensor = byte_tensor.detach().contiguous().cpu()
        try:
            return byte_tensor.numpy().tobytes()
        except (TypeError, RuntimeError):
            return bytes(byte_tensor.tolist())
    except (AttributeError, TypeError, RuntimeError):
        return None


def _read_full_source_block(
    request: XferDebugDumpRequest,
    descriptor: XferDebugDescriptor,
) -> torch.Tensor | None:
    if (
        request.pointer_kind != "source"
        or not request.config.dump_full_source_block
        or descriptor.source_block_len <= descriptor.target_block_len
    ):
        return None
    block_ptr = descriptor.src_ptr - descriptor.src_offset
    preferred_names = _preferred_tensor_names(request.pointer_kind, descriptor)
    return read_debug_bytes(
        request.cache_views,
        block_ptr,
        descriptor.source_block_len,
        preferred_names,
    )


def _preferred_tensor_names(
    pointer_kind: PointerKind,
    descriptor: XferDebugDescriptor,
) -> tuple[str, ...]:
    if pointer_kind == "source":
        return (descriptor.source_layer,)
    return descriptor.target_layers


def _build_record(
    side: str,
    pointer_kind: PointerKind,
    descriptor: XferDebugDescriptor,
    payload: torch.Tensor,
    tensor_dtype: str,
    tensor_shape: tuple[int, ...],
    tensor_name: str,
    block_stride_bytes: int,
    materialized_block_bytes: int,
    payload_padding_num_bytes: int,
    rank_tag: str | None,
    full_source_block: torch.Tensor | None,
) -> dict[str, Any]:
    payload_bytes = bytes(payload.tolist())
    record: dict[str, Any] = {
        "kind": "mooncake_xfer_debug",
        "side": side,
        "pointer_kind": pointer_kind,
        "descriptor": descriptor_to_dict(descriptor),
        "tensor_dtype": tensor_dtype,
        "tensor_shape": tensor_shape,
        "tensor_name": tensor_name,
        "block_stride_bytes": block_stride_bytes,
        "materialized_block_bytes": materialized_block_bytes,
        "rank_tag": rank_tag,
        "payload": payload,
        "payload_num_bytes": int(payload.numel()),
        "payload_materialized_num_bytes": int(
            payload.numel() - payload_padding_num_bytes
        ),
        "payload_padding_num_bytes": payload_padding_num_bytes,
        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
    }
    if full_source_block is not None:
        full_bytes = bytes(full_source_block.tolist())
        record["full_source_block"] = full_source_block
        record["full_source_block_sha256"] = hashlib.sha256(full_bytes).hexdigest()
    return record


def _record_path(
    dump_dir: Path,
    side: str,
    descriptor: XferDebugDescriptor,
    rank_tag: str | None,
    tensor_name: str,
) -> Path:
    rank_prefix = f"{_safe_name(rank_tag)}__" if rank_tag else ""
    name = (
        f"{rank_prefix}{_safe_name(side)}__tp{descriptor.tp_rank}__"
        f"tensor_{_safe_name(tensor_name)}__"
        f"{_safe_name(descriptor.transfer_id)}__"
        f"{_safe_name(descriptor.d_req_id)}__"
        f"{descriptor.descriptor_idx:05d}.pt"
    )
    return dump_dir / name


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _uint8_tensor_from_bytes(raw: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
