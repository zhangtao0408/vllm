# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.xfer_debug import (
    XferDebugCacheView,
    XferDebugDescriptor,
    XferDebugRegionInfo,
)


class KVCacheTensorLike(Protocol):
    @property
    def shared_by(self) -> Sequence[str]: ...


class KVCacheGroupLike(Protocol):
    @property
    def layer_names(self) -> Sequence[str]: ...


class KVCacheConfigLike(Protocol):
    @property
    def kv_cache_tensors(self) -> Sequence[KVCacheTensorLike]: ...

    @property
    def kv_cache_groups(self) -> Sequence[KVCacheGroupLike]: ...


class BlockRowLike(Protocol):
    def tolist(self) -> list[int]: ...


class BlockArrayLike(Protocol):
    def __getitem__(self, key: tuple[int, slice]) -> BlockRowLike: ...


class BlockTableBufferLike(Protocol):
    @property
    def np(self) -> BlockArrayLike: ...


class GroupBlockTableLike(Protocol):
    @property
    def num_blocks_per_row(self) -> Sequence[int]: ...

    @property
    def block_table(self) -> BlockTableBufferLike: ...


class MultiGroupBlockTableLike(Protocol):
    @property
    def block_tables(self) -> Sequence[GroupBlockTableLike]: ...


@dataclass(frozen=True, slots=True)
class NativeKVCacheDescriptorRequest:
    kv_cache_config: KVCacheConfigLike
    kv_caches: Mapping[str, torch.Tensor]
    request_id: str
    transfer_id: str
    tp_rank: int
    group_block_ids: tuple[tuple[int, ...], ...]


def build_xfer_debug_cache_views(
    kv_caches: Mapping[str, torch.Tensor],
) -> tuple[XferDebugCacheView, ...]:
    return tuple(
        XferDebugCacheView(
            layer_name=layer_name,
            tensor=tensor,
            base_addr=tensor.data_ptr(),
            block_len=_block_len(tensor),
        )
        for layer_name, tensor in kv_caches.items()
    )


def collect_group_block_ids(
    block_table: MultiGroupBlockTableLike,
    req_index: int,
) -> tuple[tuple[int, ...], ...]:
    group_block_ids: list[tuple[int, ...]] = []
    for group_block_table in block_table.block_tables:
        num_blocks = int(group_block_table.num_blocks_per_row[req_index])
        row = group_block_table.block_table.np[req_index, :num_blocks]
        group_block_ids.append(tuple(int(block_id) for block_id in row.tolist()))
    return tuple(group_block_ids)


def build_native_kv_cache_descriptors(
    request: NativeKVCacheDescriptorRequest,
) -> tuple[XferDebugDescriptor, ...]:
    layer_to_group_index = _build_layer_to_group_index(request.kv_cache_config)
    descriptors: list[XferDebugDescriptor] = []
    for region_idx, kv_cache_tensor in enumerate(
        request.kv_cache_config.kv_cache_tensors
    ):
        target_layers = tuple(kv_cache_tensor.shared_by)
        block_ordinal = 0
        for source_layer, group_idx in _select_region_sources(
            layer_to_group_index,
            request.kv_caches,
            target_layers,
        ):
            if group_idx >= len(request.group_block_ids):
                continue

            tensor = request.kv_caches[source_layer]
            block_len = _block_len(tensor)
            region_info = XferDebugRegionInfo(
                source_layer=source_layer,
                target_layers=target_layers,
                source_block_len=block_len,
                target_block_len=block_len,
            )
            for block_id in request.group_block_ids[group_idx]:
                ptr = tensor.data_ptr() + block_id * block_len
                descriptors.append(
                    XferDebugDescriptor(
                        transfer_id=request.transfer_id,
                        d_req_id=request.request_id,
                        tp_rank=request.tp_rank,
                        descriptor_idx=len(descriptors),
                        region_idx=region_idx,
                        block_id=block_id,
                        local_block_id=block_id,
                        remote_block_id=block_id,
                        local_block_ids=(block_id,),
                        remote_block_ids=(block_id,),
                        block_ordinals=(block_ordinal,),
                        src_ptr=ptr,
                        dst_ptr=ptr,
                        length=block_len,
                        src_offset=0,
                        dst_offset=0,
                        source_layer=source_layer,
                        target_layers=target_layers,
                        source_block_len=region_info.source_block_len,
                        target_block_len=region_info.target_block_len,
                        bucket_type=region_info.bucket_type,
                    )
                )
                block_ordinal += 1
    return tuple(descriptors)


def _build_layer_to_group_index(kv_cache_config: KVCacheConfigLike) -> dict[str, int]:
    layer_to_group_index: dict[str, int] = {}
    for group_idx, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            layer_to_group_index[layer_name] = group_idx
    return layer_to_group_index


def _select_region_sources(
    layer_to_group_index: Mapping[str, int],
    kv_caches: Mapping[str, torch.Tensor],
    target_layers: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    sources_by_group: dict[int, str] = {}
    for layer_name in target_layers:
        group_idx = layer_to_group_index.get(layer_name)
        if group_idx is None or group_idx in sources_by_group:
            continue
        if layer_name in kv_caches:
            sources_by_group[group_idx] = layer_name
    return tuple(
        (source_layer, group_idx)
        for group_idx, source_layer in sorted(sources_by_group.items())
    )


def _block_len(tensor: torch.Tensor) -> int:
    return tensor.stride(0) * tensor.element_size()
