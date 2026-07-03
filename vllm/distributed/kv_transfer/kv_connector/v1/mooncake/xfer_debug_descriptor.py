# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

ReqId = str
TransferId = str


@dataclass(frozen=True)
class XferDebugDescriptor:
    transfer_id: TransferId
    d_req_id: ReqId
    tp_rank: int
    descriptor_idx: int
    region_idx: int
    block_id: int
    local_block_id: int
    remote_block_id: int
    local_block_ids: tuple[int, ...]
    remote_block_ids: tuple[int, ...]
    block_ordinals: tuple[int, ...]
    src_ptr: int
    dst_ptr: int
    length: int
    src_offset: int
    dst_offset: int
    source_layer: str
    target_layers: tuple[str, ...]
    source_block_len: int
    target_block_len: int
    bucket_type: str


def descriptor_from_dict(raw_descriptor: Mapping[str, Any]) -> XferDebugDescriptor:
    return XferDebugDescriptor(
        transfer_id=str(raw_descriptor["transfer_id"]),
        d_req_id=str(raw_descriptor["d_req_id"]),
        tp_rank=int(raw_descriptor["tp_rank"]),
        descriptor_idx=int(raw_descriptor["descriptor_idx"]),
        region_idx=int(raw_descriptor["region_idx"]),
        block_id=int(raw_descriptor["block_id"]),
        local_block_id=int(raw_descriptor["local_block_id"]),
        remote_block_id=int(raw_descriptor["remote_block_id"]),
        local_block_ids=tuple(int(item) for item in raw_descriptor["local_block_ids"]),
        remote_block_ids=tuple(
            int(item) for item in raw_descriptor["remote_block_ids"]
        ),
        block_ordinals=tuple(int(item) for item in raw_descriptor["block_ordinals"]),
        src_ptr=int(raw_descriptor["src_ptr"]),
        dst_ptr=int(raw_descriptor["dst_ptr"]),
        length=int(raw_descriptor["length"]),
        src_offset=int(raw_descriptor["src_offset"]),
        dst_offset=int(raw_descriptor["dst_offset"]),
        source_layer=str(raw_descriptor["source_layer"]),
        target_layers=tuple(str(item) for item in raw_descriptor["target_layers"]),
        source_block_len=int(raw_descriptor["source_block_len"]),
        target_block_len=int(raw_descriptor["target_block_len"]),
        bucket_type=str(raw_descriptor["bucket_type"]),
    )


def descriptor_to_dict(descriptor: XferDebugDescriptor) -> dict[str, Any]:
    return {
        "transfer_id": descriptor.transfer_id,
        "d_req_id": descriptor.d_req_id,
        "tp_rank": descriptor.tp_rank,
        "descriptor_idx": descriptor.descriptor_idx,
        "region_idx": descriptor.region_idx,
        "block_id": descriptor.block_id,
        "local_block_id": descriptor.local_block_id,
        "remote_block_id": descriptor.remote_block_id,
        "local_block_ids": list(descriptor.local_block_ids),
        "remote_block_ids": list(descriptor.remote_block_ids),
        "block_ordinals": list(descriptor.block_ordinals),
        "src_ptr": descriptor.src_ptr,
        "dst_ptr": descriptor.dst_ptr,
        "length": descriptor.length,
        "src_offset": descriptor.src_offset,
        "dst_offset": descriptor.dst_offset,
        "source_layer": descriptor.source_layer,
        "target_layers": list(descriptor.target_layers),
        "source_block_len": descriptor.source_block_len,
        "target_block_len": descriptor.target_block_len,
        "bucket_type": descriptor.bucket_type,
    }
