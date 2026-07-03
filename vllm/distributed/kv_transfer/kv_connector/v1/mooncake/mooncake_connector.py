# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TransferTopology,
    get_current_attn_backend,
    get_current_attn_backends,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.shadow_bucket import (
    ShadowPlacement,
    bucket_slots_from_kv_cache_tensors,
    build_shadow_bucket_plan,
    canonical_cache_name,
    parse_shadow_buckets,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.stats import (
    MooncakeKVConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.xfer_debug import (
    PointerKind,
    XferDebugCacheView,
    XferDebugConfig,
    XferDebugDescriptor,
    XferDebugDumpRequest,
    XferDebugReadError,
    XferDebugRegionInfo,
    descriptor_from_dict,
    descriptor_to_dict,
    dump_xfer_debug_records,
    parse_xfer_debug_config,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec
from vllm.v1.request import RequestStatus
from vllm.v1.worker.utils import select_common_block_size

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)
KVCacheValue = torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TransferRegion:
    base_addr: int
    block_len: int
    kv_block_len: int


@dataclass(frozen=True)
class GroupTransferInfo:
    tokens_per_block: int
    blocks_per_window: int


@dataclass(frozen=True)
class KVCacheAddressMetadata:
    base_addr: int
    block_len: int
    materialized_block_len: int = 0


@dataclass(frozen=True)
class ShadowTransferSource:
    layer_name: str
    base_addr: int
    block_len: int
    kv_block_len: int
    materialized_block_len: int
    group_idx: int


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    is_kv_layout_blocks_first: bool,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert len(base_addrs) == len(block_lens), (
        "Mooncake transfer regions require matching numbers of base addresses "
        f"and block lengths, got {len(base_addrs)} and {len(block_lens)}."
    )
    regions: list[TransferRegion] = []
    for base_addr, block_len in zip(base_addrs, block_lens):
        kv_block_len = block_len // 2 if is_kv_layout_blocks_first else block_len
        regions.append(
            TransferRegion(
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
            )
        )
        if is_kv_layout_blocks_first:
            regions.append(
                TransferRegion(
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, remote_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _format_transfer_region_sample(
    regions: list[TransferRegion], limit: int = 8
) -> str:
    sample = [
        (idx, region.base_addr, region.block_len, region.kv_block_len)
        for idx, region in enumerate(regions[:limit])
    ]
    suffix = "" if len(regions) <= limit else f" ... +{len(regions) - limit} more"
    return f"count={len(regions)} sample={sample}{suffix}"


def _format_transfer_descriptor_sample(
    src_ptrs: list[int],
    dst_ptrs: list[int],
    lengths: list[int],
    limit: int = 16,
) -> str:
    if not src_ptrs or not dst_ptrs or not lengths:
        return (
            f"src_count={len(src_ptrs)} dst_count={len(dst_ptrs)} "
            f"len_count={len(lengths)} total_bytes={sum(lengths)} sample=[]"
        )

    descriptor_count = min(len(src_ptrs), len(dst_ptrs), len(lengths))
    sample = [
        (
            idx,
            src_ptr,
            dst_ptr,
            length,
            src_ptr + length,
            dst_ptr + length,
        )
        for idx, (src_ptr, dst_ptr, length) in enumerate(
            zip(src_ptrs[:limit], dst_ptrs[:limit], lengths[:limit])
        )
    ]
    src_start = min(src_ptrs[:descriptor_count])
    src_end = max(
        src_ptr + length
        for src_ptr, length in zip(
            src_ptrs[:descriptor_count], lengths[:descriptor_count]
        )
    )
    dst_start = min(dst_ptrs[:descriptor_count])
    dst_end = max(
        dst_ptr + length
        for dst_ptr, length in zip(
            dst_ptrs[:descriptor_count], lengths[:descriptor_count]
        )
    )
    suffix = (
        "" if descriptor_count <= limit else f" ... +{descriptor_count - limit} more"
    )
    return (
        f"src_count={len(src_ptrs)} dst_count={len(dst_ptrs)} "
        f"len_count={len(lengths)} total_bytes={sum(lengths)} "
        f"src_range=({src_start}, {src_end}) dst_range=({dst_start}, {dst_end}) "
        f"sample=(idx, src, dst, len, src_end, dst_end) {sample}{suffix}"
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        logger.warning(
            "[PD_DEBUG][region_count_mismatch] "
            "local_count=%d remote_count=%d local_tp_size=%d "
            "remote_tp_size=%d producer_cache_replicated=%s "
            "local_regions=%s remote_regions=%s",
            len(local_regions),
            len(remote_regions),
            local_tp_size,
            remote_tp_size,
            producer_cache_replicated,
            [(region.base_addr, region.kv_block_len) for region in local_regions],
            [(region.base_addr, region.kv_block_len) for region in remote_regions],
        )
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


def _get_tensor_storage_key(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        try:
            return tensor.storage().data_ptr()
        except (AttributeError, RuntimeError):
            return tensor.data_ptr()


def _merge_register_ranges(
    ranges_by_storage: dict[int, list[tuple[int, int]]],
) -> tuple[list[int], list[int]]:
    ptrs: list[int] = []
    lengths: list[int] = []
    register_merge_gap_bytes = 4096

    for ranges in ranges_by_storage.values():
        ranges.sort(key=lambda item: item[0])
        merged_start, merged_end = ranges[0]
        for start, end in ranges[1:]:
            if start <= merged_end + register_merge_gap_bytes:
                merged_end = max(merged_end, end)
                continue
            ptrs.append(merged_start)
            lengths.append(merged_end - merged_start)
            merged_start, merged_end = start, end
        ptrs.append(merged_start)
        lengths.append(merged_end - merged_start)

    return ptrs, lengths


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[list[int]]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    kv_cache_group_layer_names: list[list[str]] = msgspec.field(default_factory=list)
    kv_cache_group_block_lens: list[int] = msgspec.field(default_factory=list)
    kv_cache_layer_materialized_block_lens: dict[str, int] = msgspec.field(
        default_factory=dict
    )


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None
    debug_descriptors: list[dict[str, Any]] | None = None


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[list[int]]
    ready: asyncio.Event
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[list[int]]]] = {}
        self.reqs_not_processed: set[TransferId] = set()

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[list[int]],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
            )
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)


class MooncakeConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            assert (
                kv_cache_config is not None
            ), "kv_cache_config is required for SCHEDULER role"
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id, kv_cache_config)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            # This fallback mostly exists for unit tests that instantiate the
            # connector without a fully populated model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to HND for "
            "heterogeneous TP-safe KV transfer."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, (block_ids,))

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        pass

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return worker-local transfer stats since the last call.

        Note the P/D asymmetry: because Mooncake is P-push (P calls
        batch_transfer_sync_write), P records successful transfer latency,
        bytes, and descriptor counts, while D only records failures
        (recv/ZMQ errors). Aggregated NIXL-style dashboards will find
        successful-transfer metrics on the P worker, not D.
        """
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return MooncakeKVConnectorStats(data=data or {})


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        self.vllm_config = vllm_config
        self.block_size = vllm_config.cache_config.block_size
        self.kv_cache_config = kv_cache_config

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        self._is_hma_required = (
            not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and any(
                not isinstance(g.kv_cache_spec, FullAttentionSpec)
                for g in kv_cache_config.kv_cache_groups
            )
        )

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[list[int]]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()

        self.group_transfer_info = [
            self._get_group_transfer_info(group)
            for group in kv_cache_config.kv_cache_groups
        ]
        self.prefill_compress_ratio = self._get_prefill_compress_ratio()
        self.blocks_per_sw = [
            group_info.blocks_per_window for group_info in self.group_transfer_info
        ]

    def get_sw_clipped_blocks(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
    ) -> list[list[int]]:
        """Clip per-group block IDs to sliding window size."""
        if len(block_ids) == 0 or not self._is_hma_required:
            return list(block_ids)
        return [
            blocks[-self.blocks_per_sw[i] :] if self.blocks_per_sw[i] > 0 else blocks
            for i, blocks in enumerate(block_ids)
        ]

    @staticmethod
    def _get_group_unique_specs(group: Any) -> list[Any]:
        group_spec = group.kv_cache_spec
        inner_specs = getattr(group_spec, "kv_cache_specs", None)
        if not isinstance(inner_specs, dict):
            return [group_spec]

        specs = []
        for layer_name in group.layer_names:
            layer_spec = inner_specs.get(layer_name)
            if layer_spec is not None and layer_spec not in specs:
                specs.append(layer_spec)
        return specs or [group_spec]

    def _get_prefill_compress_ratio(self) -> int:
        hf_config = getattr(self.vllm_config.model_config, "hf_config", None)
        compress_ratios = getattr(hf_config, "compress_ratios", None)
        if isinstance(compress_ratios, dict):
            ratio_values = compress_ratios.values()
        elif isinstance(compress_ratios, (list, tuple)):
            ratio_values = compress_ratios
        else:
            return 1
        positive_ratios = [
            int(ratio)
            for ratio in ratio_values
            if isinstance(ratio, int) and ratio > 1
        ]
        return max(positive_ratios, default=1)

    def _get_group_transfer_info(self, group: Any) -> GroupTransferInfo:
        specs = self._get_group_unique_specs(group)
        first_spec = specs[0] if specs else group.kv_cache_spec
        block_size = getattr(
            group.kv_cache_spec,
            "block_size",
            getattr(first_spec, "block_size", self.block_size),
        )
        if block_size <= 0:
            block_size = self.block_size

        sliding_window = 0
        for spec in specs:
            if isinstance(spec, SlidingWindowSpec):
                sliding_window = max(sliding_window, spec.sliding_window)

        return GroupTransferInfo(
            tokens_per_block=block_size,
            blocks_per_window=cdiv(sliding_window, block_size) + 1
            if sliding_window
            else 0,
        )

    def _get_group_transfer_block_size(self, group_idx: int) -> int:
        if group_idx < len(self.group_transfer_info):
            return self.group_transfer_info[group_idx].tokens_per_block
        if group_idx >= len(self.kv_cache_config.kv_cache_groups):
            return self.block_size
        spec = self.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
        group_block_size = getattr(spec, "block_size", self.block_size)
        return group_block_size if group_block_size > 0 else self.block_size

    def _state_prefill_token_count(self, num_prompt_tokens: int) -> int:
        if self.prefill_compress_ratio <= 1 or num_prompt_tokens <= 1:
            return num_prompt_tokens
        prefill_tokens = num_prompt_tokens - 1
        return (
            prefill_tokens // self.prefill_compress_ratio
        ) * self.prefill_compress_ratio

    def _clip_blocks_to_external_tokens(
        self,
        block_ids: tuple[list[int], ...] | list[list[int]],
        num_external_tokens: int,
    ) -> list[list[int]]:
        if len(block_ids) == 0 or num_external_tokens <= 0:
            return [[] for _ in range(len(block_ids))]
        clipped_blocks: list[list[int]] = []
        for group_idx, blocks in enumerate(block_ids):
            group_block_size = self._get_group_transfer_block_size(group_idx)
            num_external_blocks = cdiv(num_external_tokens, group_block_size)
            clipped_blocks.append(list(blocks[:num_external_blocks]))
        return clipped_blocks

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            # Remote prefill: get all prompt blocks from remote.
            assert not self.is_kv_producer
            token_ids = request.prompt_token_ids or []
            external_token_count = self._state_prefill_token_count(len(token_ids))
            count = max(external_token_count - num_computed_tokens, 0)
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                unhashed_block_ids = (
                    blocks.get_unhashed_block_ids_all_groups()
                    if num_external_tokens > 0
                    else ()
                )
                external_block_ids = self._clip_blocks_to_external_tokens(
                    unhashed_block_ids, num_external_tokens
                )
                local_block_ids = self.get_sw_clipped_blocks(external_block_ids)
                logger.debug(
                    "MooncakeConnector recv blocks: req_id=%s "
                    "num_external_tokens=%s raw_group_lens=%s "
                    "external_group_lens=%s clipped_group_lens=%s "
                    "tokens_per_block=%s",
                    request.request_id,
                    num_external_tokens,
                    [len(group) for group in unhashed_block_ids],
                    [len(group) for group in external_block_ids],
                    [len(group) for group in local_block_ids],
                    [
                        group_info.tokens_per_block
                        for group_info in self.group_transfer_info
                    ],
                )
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            return False, None

        if not params.get("do_remote_decode"):
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = any(len(group) > 0 for group in block_ids)

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (
                request,
                self.get_sw_clipped_blocks(block_ids),
            )

        return delay_free_blocks, None


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        engine_id: str,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config
        # Capture device BEFORE TransferEngine init — MNNVL's NVLink allocator
        # may change the current CUDA device during engine.initialize().
        self.device_id = torch.accelerator.current_device_index()
        current_platform.set_device(self.device_id)

        self.engine = TransferEngine()
        self.hostname = get_ip()
        self.shadow_block_len_per_layer: list[int] = []
        self.shadow_bucket_plan: tuple[ShadowPlacement, ...] = ()
        self.shadow_sources_by_layer: dict[str, ShadowTransferSource] = {}
        self.kv_cache_group_layer_names: list[list[str]] = []
        self.kv_cache_group_block_lens: list[int] = []
        self.kv_cache_layer_materialized_block_lens: dict[str, int] = {}
        self.layer_to_group_index: dict[str, int] = {}

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.xfer_debug_config: XferDebugConfig | None = parse_xfer_debug_config(
            kv_transfer_config.kv_connector_extra_config.get("xfer_debug")
        )
        self._xfer_debug_cache_views: list[XferDebugCacheView] = []
        self._xfer_debug_dumped_transfers: set[TransferId] = set()
        self.kv_role: str = kv_transfer_config.kv_role
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        
        mlx_bond_list = os.getenv("MLX_BONDS", "mlx5_bond_4")

        ret_value = self.engine.initialize(
            self.hostname, "P2PHANDSHAKE", protocol, mlx_bond_list
        )
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0
        self.block_len_per_layer: list[int] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reqs_need_send: dict[TransferId, SendBlockMeta] = {}

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            # Each pool thread must be bound to the correct CUDA device
            # because CUDA device selection is thread-local.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
                initializer=self._bind_sender_thread_device,
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()

        self.xfer_stats = MooncakeKVConnectorStats()

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.kv_cache_config = kv_cache_config
        self.use_mla = self.model_config.use_mla
        self._sync_block_size_with_kernel()

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self.transfer_topo = TransferTopology(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            block_size=self.block_size,
            engine_id=self.engine_id,
            is_mla=self.use_mla,
            is_mamba=False,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=[backend],
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    def _sync_block_size_with_kernel(self) -> None:
        # When speculative decoding (e.g. Eagle) is enabled, the main model
        # and draft model may use different attention backends with different
        # physical block sizes. Pick the common (smallest) block size so that
        # KV-cache registration and transfer work correctly for both models.
        backends = get_current_attn_backends(self.vllm_config)
        kernel_block_size = select_common_block_size(self.block_size, backends)
        if self.block_size != kernel_block_size:
            logger.info_once(
                "User-specified logical block size (%s) does not match"
                " physical kernel block size (%s). Using the latter.",
                self.block_size,
                kernel_block_size,
            )
            assert self.block_size > kernel_block_size
            self.block_size = kernel_block_size

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config) and hasattr(
                self, "bootstrap_server"
            ):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr, meta.block_lens
        )

        if self.shadow_bucket_plan:
            local_regions = self._get_shadow_transfer_regions()
            validation_err = self._validate_shadow_transfer_regions(
                local_regions, remote_regions
            )
        else:
            local_regions = self._get_transfer_regions(
                self.kv_caches_base_addr, self.block_len_per_layer
            )
            validation_err = _validate_asymmetric_region_lengths(
                local_regions=local_regions,
                remote_regions=remote_regions,
                local_tp_size=self.tp_size,
                remote_tp_size=meta.remote_tp_size,
                producer_cache_replicated=self._producer_cache_is_replicated(),
            )
        logger.debug(
            "[PD_DEBUG][regions_before_validate] "
            "role=%s local_tp_rank=%s local_tp_size=%s remote_tp_rank=%s "
            "remote_tp_size=%s local_region_count=%d remote_region_count=%d "
            "local_block_lens=%s remote_block_lens=%s local_base_count=%d "
            "remote_base_count=%d producer_cache_replicated=%s",
            self.kv_role,
            self.tp_rank,
            self.tp_size,
            meta.remote_tp_rank,
            meta.remote_tp_size,
            len(local_regions),
            len(remote_regions),
            [region.kv_block_len for region in local_regions],
            [region.kv_block_len for region in remote_regions],
            len(self.kv_caches_base_addr),
            len(meta.kv_caches_base_addr),
            self._producer_cache_is_replicated(),
        )
        if validation_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=validation_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            if transfer_id not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[transfer_id] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                )
            send_meta = self.reqs_need_send[transfer_id]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                if send_meta.transfer_id in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            (
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
                debug_descriptors,
            ) = await self._build_transfer_params(
                ready_reqs,
                meta,
                local_regions,
                remote_regions,
            )
            err_req_set = set(err_reqs)
            ok_ready_reqs = [
                (d_req_id, send_meta)
                for d_req_id, send_meta in ready_reqs
                if d_req_id not in err_req_set
            ]

            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                self._dump_xfer_debug_records(
                    side="producer",
                    descriptors=debug_descriptors,
                    pointer_kind="source",
                )
                if self.shadow_bucket_plan:
                    logger.debug(
                        "[PD_DEBUG][shadow_transfer_before_send] "
                        "remote_session=%s local_regions=%s remote_regions=%s "
                        "descriptors=%s",
                        remote_session,
                        _format_transfer_region_sample(local_regions),
                        _format_transfer_region_sample(remote_regions),
                        _format_transfer_descriptor_sample(src_ptrs, dst_ptrs, lengths),
                    )
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    transfer_err_msg = f"Mooncake transfer engine returned {ret_value}"
                    err_msg = (
                        transfer_err_msg
                        if err_msg is None
                        else f"{err_msg}; {transfer_err_msg}"
                    )
                    err_reqs = list(err_reqs)
                    for d_req_id, _ in ok_ready_reqs:
                        err_reqs.append(d_req_id)
                        err_req_set.add(d_req_id)
                    ok_ready_reqs = []

            for d_req_id, send_meta in ready_reqs:
                send_meta.sending -= 1

                if d_req_id in err_req_set:
                    continue

                send_meta.sent += 1
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_meta.transfer_id, None) is not None
                ):
                    self.finished_sending_reqs.add(send_meta.p_req_id)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                err_reqs=err_reqs or None,
                err_msg=err_msg,
                debug_descriptors=self._encode_xfer_debug_descriptors(
                    debug_descriptors, ok_ready_reqs
                ),
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: %s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    def _xfer_debug_transfer_enabled(self, transfer_id: TransferId) -> bool:
        config = self.xfer_debug_config
        if config is None or config.max_requests == 0:
            return False
        if transfer_id in self._xfer_debug_dumped_transfers:
            return True
        if len(self._xfer_debug_dumped_transfers) >= config.max_requests:
            return False
        self._xfer_debug_dumped_transfers.add(transfer_id)
        return True

    def _filter_xfer_debug_descriptors(
        self, descriptors: list[XferDebugDescriptor]
    ) -> tuple[XferDebugDescriptor, ...]:
        return tuple(
            descriptor
            for descriptor in descriptors
            if self._xfer_debug_transfer_enabled(descriptor.transfer_id)
        )

    def _dump_xfer_debug_records(
        self,
        side: str,
        descriptors: list[XferDebugDescriptor],
        pointer_kind: PointerKind,
    ) -> None:
        config = self.xfer_debug_config
        if config is None or not descriptors:
            return
        selected_descriptors = self._filter_xfer_debug_descriptors(descriptors)
        if not selected_descriptors:
            return
        try:
            dumped = dump_xfer_debug_records(
                XferDebugDumpRequest(
                    config=config,
                    cache_views=tuple(self._xfer_debug_cache_views),
                    side=side,
                    pointer_kind=pointer_kind,
                    descriptors=selected_descriptors,
                    rank_tag=f"dp{self.dp_rank}__tp{self.tp_rank}",
                )
            )
        except (OSError, RuntimeError, XferDebugReadError, ValueError) as exc:
            logger.warning("Mooncake xfer debug dump failed on %s side: %s", side, exc)
            return
        logger.info(
            "Mooncake xfer debug dumped %d descriptor record(s) to %s on %s side.",
            dumped,
            config.dump_dir,
            side,
        )

    def _encode_xfer_debug_descriptors(
        self,
        descriptors: list[XferDebugDescriptor],
        ok_ready_reqs: list[tuple[ReqId, SendBlockMeta]],
    ) -> list[dict[str, Any]] | None:
        if self.xfer_debug_config is None or not descriptors:
            return None
        ok_req_ids = {d_req_id for d_req_id, _ in ok_ready_reqs}
        encoded = [
            descriptor_to_dict(descriptor)
            for descriptor in descriptors
            if descriptor.d_req_id in ok_req_ids
            and descriptor.transfer_id in self._xfer_debug_dumped_transfers
        ]
        return encoded or None

    def _get_xfer_debug_region_info(
        self,
        region_idx: int,
        local_region: TransferRegion,
        remote_region: TransferRegion,
    ) -> XferDebugRegionInfo:
        base_region_idx = (
            region_idx // 2
            if self.transfer_topo.virtually_split_kv_in_blocks
            else region_idx
        )
        if self.shadow_bucket_plan and base_region_idx < len(self.shadow_bucket_plan):
            placement = self.shadow_bucket_plan[base_region_idx]
            return XferDebugRegionInfo(
                source_layer=placement.source.layer_name,
                target_layers=placement.target_layer_names,
                source_block_len=local_region.kv_block_len,
                target_block_len=remote_region.kv_block_len,
            )
        if self.kv_cache_config is not None and base_region_idx < len(
            self.kv_cache_config.kv_cache_tensors
        ):
            target_layers = tuple(
                self.kv_cache_config.kv_cache_tensors[base_region_idx].shared_by
            )
        else:
            target_layers = (f"region_{base_region_idx}",)
        return XferDebugRegionInfo(
            source_layer=target_layers[0],
            target_layers=target_layers,
            source_block_len=local_region.kv_block_len,
            target_block_len=remote_region.kv_block_len,
        )

    def _make_xfer_debug_descriptor(
        self,
        transfer_id: TransferId,
        d_req_id: ReqId,
        descriptor_idx: int,
        region_idx: int,
        local_region: TransferRegion,
        remote_region: TransferRegion,
        src_ptr: int,
        dst_ptr: int,
        length: int,
        src_region_offset: int,
        dst_region_offset: int,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        block_ordinals: list[int],
    ) -> XferDebugDescriptor:
        region_info = self._get_xfer_debug_region_info(
            region_idx, local_region, remote_region
        )
        return XferDebugDescriptor(
            transfer_id=transfer_id,
            d_req_id=d_req_id,
            tp_rank=self.tp_rank,
            descriptor_idx=descriptor_idx,
            region_idx=region_idx,
            block_id=remote_block_ids[0],
            local_block_id=local_block_ids[0],
            remote_block_id=remote_block_ids[0],
            local_block_ids=tuple(local_block_ids),
            remote_block_ids=tuple(remote_block_ids),
            block_ordinals=tuple(block_ordinals),
            src_ptr=src_ptr,
            dst_ptr=dst_ptr,
            length=length,
            src_offset=src_region_offset,
            dst_offset=dst_region_offset,
            source_layer=region_info.source_layer,
            target_layers=tuple(region_info.target_layers),
            source_block_len=region_info.source_block_len,
            target_block_len=region_info.target_block_len,
            bucket_type=region_info.bucket_type,
        )

    def _flatten_shadow_block_ids(
        self,
        d_req_id: ReqId,
        local_block_ids_per_group: list[list[int]],
        remote_block_ids_per_group: list[list[int]],
    ) -> tuple[list[int], list[int], str | None]:
        local_block_ids = max(local_block_ids_per_group, key=len, default=[])
        remote_block_ids = max(remote_block_ids_per_group, key=len, default=[])
        local_group_lens = [len(group) for group in local_block_ids_per_group]
        remote_group_lens = [len(group) for group in remote_block_ids_per_group]

        if len(local_block_ids) < len(remote_block_ids):
            return (
                [],
                [],
                "P num blocks less than D "
                f"(shadow local_lens={local_group_lens} "
                f"remote_lens={remote_group_lens})",
            )
        if len(local_block_ids) > len(remote_block_ids):
            local_block_ids = local_block_ids[-len(remote_block_ids) :]

        logger.debug(
            "req %s: shadow bucket selected canonical KV block groups from "
            "local_lens=%s remote_lens=%s to %d transfer blocks.",
            d_req_id,
            local_group_lens,
            remote_group_lens,
            len(remote_block_ids),
        )
        return list(local_block_ids), list(remote_block_ids), None

    def _get_remote_layer_to_group_index(
        self,
        group_layer_names: list[list[str]],
    ) -> dict[str, int]:
        layer_to_group: dict[str, int] = {}
        for group_idx, layer_names in enumerate(group_layer_names):
            for layer_name in layer_names:
                layer_to_group[canonical_cache_name(layer_name)] = group_idx
        return layer_to_group

    def _trim_shadow_group_blocks(
        self,
        local_group: list[int],
        remote_group: list[int],
        d_req_id: ReqId,
        source_layer: str,
        target_layer: str,
    ) -> tuple[list[int], list[int], str | None]:
        if len(local_group) < len(remote_group):
            return (
                [],
                [],
                "P num blocks less than D "
                f"(req={d_req_id} source={source_layer} target={target_layer} "
                f"local_len={len(local_group)} remote_len={len(remote_group)} "
                f"local_tail={local_group[-4:]} remote_tail={remote_group[-4:]})",
            )
        if len(local_group) > len(remote_group):
            local_group = local_group[-len(remote_group) :]
        return list(local_group), list(remote_group), None

    def _get_shadow_mla_reorder_plan(
        self,
        source: ShadowTransferSource,
        remote_region: TransferRegion,
        target_block_len: int,
    ) -> tuple[int, int] | None:
        if (
            source.block_len == 40960
            and remote_region.block_len == 37440
            and target_block_len == 37376
        ):
            return 0, 64
        if (
            source.block_len == 40960
            and remote_region.block_len == 1728
            and target_block_len == 1168
        ):
            return 62, 2
        return None

    def _append_shadow_copy_descriptor(
        self,
        *,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        debug_descriptors: list[XferDebugDescriptor],
        transfer_id: TransferId,
        d_req_id: ReqId,
        region_idx: int,
        source: ShadowTransferSource,
        target_layer_name: str,
        src_ptr: int,
        dst_ptr: int,
        length: int,
        src_region_offset: int,
        dst_region_offset: int,
        local_block_id: int,
        remote_block_id: int,
        block_ordinal: int,
        target_block_len: int,
    ) -> None:
        if self.xfer_debug_config is not None:
            region_info = XferDebugRegionInfo(
                source_layer=source.layer_name,
                target_layers=(target_layer_name,),
                source_block_len=source.materialized_block_len,
                target_block_len=target_block_len,
            )
            debug_descriptors.append(
                XferDebugDescriptor(
                    transfer_id=transfer_id,
                    d_req_id=d_req_id,
                    tp_rank=self.tp_rank,
                    descriptor_idx=len(src_ptrs),
                    region_idx=region_idx,
                    block_id=remote_block_id,
                    local_block_id=local_block_id,
                    remote_block_id=remote_block_id,
                    local_block_ids=(local_block_id,),
                    remote_block_ids=(remote_block_id,),
                    block_ordinals=(block_ordinal,),
                    src_ptr=src_ptr,
                    dst_ptr=dst_ptr,
                    length=length,
                    src_offset=src_region_offset,
                    dst_offset=dst_region_offset,
                    source_layer=source.layer_name,
                    target_layers=(target_layer_name,),
                    source_block_len=source.materialized_block_len,
                    target_block_len=target_block_len,
                    bucket_type=region_info.bucket_type,
                )
            )
        src_ptrs.append(src_ptr)
        dst_ptrs.append(dst_ptr)
        lengths.append(length)

    def _append_shadow_block_transfers(
        self,
        *,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        debug_descriptors: list[XferDebugDescriptor],
        transfer_id: TransferId,
        d_req_id: ReqId,
        region_idx: int,
        source: ShadowTransferSource,
        target_layer_name: str,
        remote_region: TransferRegion,
        local_block_ids: list[int],
        remote_block_ids: list[int],
        target_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> str | None:
        should_transfer, src_offset, dst_offset, transfer_len = (
            self._get_sender_transfer_plan(
                local_kv_block_len=source.kv_block_len,
                remote_kv_block_len=remote_region.kv_block_len,
                remote_tp_rank=remote_tp_rank,
                remote_tp_size=remote_tp_size,
            )
        )
        if not should_transfer:
            return None

        mla_reorder_plan = self._get_shadow_mla_reorder_plan(
            source, remote_region, target_block_len
        )
        if mla_reorder_plan is not None:
            if src_offset != 0 or dst_offset != 0:
                return (
                    "Mooncake shadow MLA reorder transfer does not support TP "
                    "offsets."
                )
            start_row, rows = mla_reorder_plan
            src_row_len = 640
            dst_data_row_len = 576
            dst_scale_row_len = 8
            dst_scale_base = rows * dst_data_row_len
            # A5 stores each 640-byte MLA row as:
            #   [128B rope bf16][448B nope fp8][8B UE8M0 scales][56B pad].
            # H20 fp8_ds_mla consumes each block as:
            #   data plane [rows][448B nope fp8 + 128B rope bf16],
            #   then scale plane [rows][8B UE8M0 scales].
            row_segments = (
                (128, 0, 448, False),
                (0, 448, 128, False),
                (576, 0, dst_scale_row_len, True),
            )
            for block_ordinal, (local_block_id, remote_block_id) in enumerate(
                zip(local_block_ids, remote_block_ids)
            ):
                for row in range(start_row, start_row + rows):
                    dst_row = row - start_row
                    for src_in_row, dst_in_row, length, is_scale in row_segments:
                        src_region_offset = row * src_row_len + src_in_row
                        if is_scale:
                            dst_region_offset = (
                                dst_scale_base + dst_row * dst_scale_row_len
                            )
                        else:
                            dst_region_offset = (
                                dst_row * dst_data_row_len + dst_in_row
                            )
                        self._append_shadow_copy_descriptor(
                            src_ptrs=src_ptrs,
                            dst_ptrs=dst_ptrs,
                            lengths=lengths,
                            debug_descriptors=debug_descriptors,
                            transfer_id=transfer_id,
                            d_req_id=d_req_id,
                            region_idx=region_idx,
                            source=source,
                            target_layer_name=target_layer_name,
                            src_ptr=(
                                source.base_addr
                                + local_block_id * source.block_len
                                + src_region_offset
                            ),
                            dst_ptr=(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            ),
                            length=length,
                            src_region_offset=src_region_offset,
                            dst_region_offset=dst_region_offset,
                            local_block_id=local_block_id,
                            remote_block_id=remote_block_id,
                            block_ordinal=block_ordinal,
                            target_block_len=target_block_len,
                        )
            return None

        copy_len = min(
            transfer_len,
            source.materialized_block_len,
            target_block_len,
            source.kv_block_len - src_offset,
            remote_region.kv_block_len - dst_offset,
        )
        for block_ordinal, (local_block_id, remote_block_id) in enumerate(
            zip(local_block_ids, remote_block_ids)
        ):
            self._append_shadow_copy_descriptor(
                src_ptrs=src_ptrs,
                dst_ptrs=dst_ptrs,
                lengths=lengths,
                debug_descriptors=debug_descriptors,
                transfer_id=transfer_id,
                d_req_id=d_req_id,
                region_idx=region_idx,
                source=source,
                target_layer_name=target_layer_name,
                src_ptr=(
                    source.base_addr + local_block_id * source.block_len + src_offset
                ),
                dst_ptr=(
                    remote_region.base_addr
                    + remote_block_id * remote_region.block_len
                    + dst_offset
                ),
                length=copy_len,
                src_region_offset=src_offset,
                dst_region_offset=dst_offset,
                local_block_id=local_block_id,
                remote_block_id=remote_block_id,
                block_ordinal=block_ordinal,
                target_block_len=target_block_len,
            )
        return None

    def _append_shadow_transfer_params(
        self,
        *,
        d_req_id: ReqId,
        send_meta: SendBlockMeta,
        remote_block_ids_per_group: list[list[int]],
        agent_meta: MooncakeXferMetadata,
        remote_regions: list[TransferRegion],
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
        debug_descriptors: list[XferDebugDescriptor],
    ) -> str | None:
        remote_layer_to_group_index = self._get_remote_layer_to_group_index(
            agent_meta.kv_cache_group_layer_names
        )
        if not remote_layer_to_group_index:
            return "Mooncake shadow bucket requires remote KV group metadata."

        for region_idx, (placement, remote_region) in enumerate(
            zip(self.shadow_bucket_plan, remote_regions)
        ):
            for layer_mapping in placement.layer_mappings:
                source_key = canonical_cache_name(layer_mapping.source.layer_name)
                target_key = canonical_cache_name(layer_mapping.target_layer_name)
                source = self.shadow_sources_by_layer.get(source_key)
                remote_group_idx = remote_layer_to_group_index.get(target_key)
                if source is None or remote_group_idx is None:
                    continue
                if source.group_idx >= len(send_meta.local_block_ids):
                    continue
                if remote_group_idx >= len(remote_block_ids_per_group):
                    continue
                target_block_len = (
                    agent_meta.kv_cache_layer_materialized_block_lens.get(target_key)
                )
                if target_block_len is None:
                    target_block_len = (
                        agent_meta.kv_cache_group_block_lens[remote_group_idx]
                        if remote_group_idx < len(agent_meta.kv_cache_group_block_lens)
                        else remote_region.kv_block_len
                    )
                local_block_ids, remote_block_ids, block_error = (
                    self._trim_shadow_group_blocks(
                        send_meta.local_block_ids[source.group_idx],
                        remote_block_ids_per_group[remote_group_idx],
                        d_req_id,
                        source.layer_name,
                        layer_mapping.target_layer_name,
                    )
                )
                if block_error is not None:
                    return block_error
                if not local_block_ids:
                    continue
                append_error = self._append_shadow_block_transfers(
                    src_ptrs=src_ptrs,
                    dst_ptrs=dst_ptrs,
                    lengths=lengths,
                    debug_descriptors=debug_descriptors,
                    transfer_id=send_meta.transfer_id,
                    d_req_id=d_req_id,
                    region_idx=region_idx,
                    source=source,
                    target_layer_name=layer_mapping.target_layer_name,
                    remote_region=remote_region,
                    local_block_ids=local_block_ids,
                    remote_block_ids=remote_block_ids,
                    target_block_len=target_block_len,
                    remote_tp_rank=agent_meta.remote_tp_rank,
                    remote_tp_size=agent_meta.remote_tp_size,
                )
                if append_error is not None:
                    return append_error
        return None

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[
        list[int],
        list[int],
        list[int],
        list[ReqId],
        str | None,
        list[XferDebugDescriptor],
    ]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        debug_descriptors: list[XferDebugDescriptor] = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids_per_group = agent_meta.req_blocks[d_req_id]

            if not remote_block_ids_per_group or all(
                len(g) == 0 for g in remote_block_ids_per_group
            ):
                continue

            if self.shadow_bucket_plan and agent_meta.kv_cache_group_layer_names:
                shadow_error = self._append_shadow_transfer_params(
                    d_req_id=d_req_id,
                    send_meta=send_meta,
                    remote_block_ids_per_group=remote_block_ids_per_group,
                    agent_meta=agent_meta,
                    remote_regions=remote_regions,
                    src_ptrs=src_ptrs,
                    dst_ptrs=dst_ptrs,
                    lengths=lengths,
                    debug_descriptors=debug_descriptors,
                )
                if shadow_error is not None:
                    logger.error("req %s: %s", d_req_id, shadow_error)
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = shadow_error
                continue

            # Per-group partial hit trimming, then flatten.
            # With HMA, groups share the same KV tensor but use different
            # block ranges.  We trim and concatenate so the coalescer and
            # address math see one flat block list — same as non-HMA, but
            # now including blocks from every group.
            local_block_ids: list[int] = []
            remote_block_ids: list[int] = []
            has_block_error = False
            if len(send_meta.local_block_ids) != len(remote_block_ids_per_group):
                if self.shadow_bucket_plan:
                    local_block_ids, remote_block_ids, block_error = (
                        self._flatten_shadow_block_ids(
                            d_req_id,
                            send_meta.local_block_ids,
                            remote_block_ids_per_group,
                        )
                    )
                    if block_error is not None:
                        logger.error("req %s: %s", d_req_id, block_error)
                        err_reqs.append(d_req_id)
                        if err_msg is None:
                            err_msg = block_error
                        continue
                    if not local_block_ids:
                        continue
                else:
                    logger.error(
                        "req %s: KV group count mismatch: local=%d, remote=%d",
                        d_req_id,
                        len(send_meta.local_block_ids),
                        len(remote_block_ids_per_group),
                    )
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = "KV group count mismatch"
                    continue
            else:
                for local_group, remote_group in zip(
                    send_meta.local_block_ids, remote_block_ids_per_group
                ):
                    n_local = len(local_group)
                    n_remote = len(remote_group)
                    if n_local < n_remote:
                        logger.error(
                            "req %s: local blocks(%d) < remote blocks(%d) "
                            "in a KV cache group",
                            d_req_id,
                            n_local,
                            n_remote,
                        )
                        has_block_error = True
                        break
                    if n_local > n_remote:
                        # Partial prefix cache hit: just read uncomputed blocks.
                        local_group = local_group[-n_remote:]
                    local_block_ids.extend(local_group)
                    remote_block_ids.extend(remote_group)

                if has_block_error:
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = "P num blocks less than D"
                    continue

            if not local_block_ids:
                continue

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for region_idx, (local_region, remote_region) in enumerate(
                zip(local_regions, remote_regions)
            ):
                should_transfer, src_region_offset, dst_region_offset, transfer_len = (
                    self._get_sender_transfer_plan(
                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=agent_meta.remote_tp_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                if self.shadow_bucket_plan:
                    transfer_len = self._get_shadow_copy_len(
                        local_region=local_region,
                        remote_region=remote_region,
                        src_region_offset=src_region_offset,
                        dst_region_offset=dst_region_offset,
                        transfer_len=transfer_len,
                    )

                assert (
                    src_region_offset + transfer_len <= local_region.kv_block_len
                ), "Computed source transfer region exceeds local KV block size."
                assert (
                    dst_region_offset + transfer_len <= remote_region.kv_block_len
                ), "Computed destination transfer region exceeds remote KV block size."
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                can_coalesce = _can_coalesce_block_transfers(
                    local_region_block_len=local_region.block_len,
                    remote_region_block_len=remote_region.block_len,
                    src_region_offset=src_region_offset,
                    dst_region_offset=dst_region_offset,
                    transfer_len=transfer_len,
                )

                block_ordinal_cursor = 0
                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    block_ordinals = list(
                        range(
                            block_ordinal_cursor,
                            block_ordinal_cursor + len(group_local_block_id),
                        )
                    )
                    block_ordinal_cursor += len(group_local_block_id)
                    if can_coalesce:
                        src_ptr = (
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptr = (
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        length = transfer_len * len(group_local_block_id)
                        if self.xfer_debug_config is not None:
                            debug_descriptors.append(
                                self._make_xfer_debug_descriptor(
                                    transfer_id=send_meta.transfer_id,
                                    d_req_id=d_req_id,
                                    descriptor_idx=len(src_ptrs),
                                    region_idx=region_idx,
                                    local_region=local_region,
                                    remote_region=remote_region,
                                    src_ptr=src_ptr,
                                    dst_ptr=dst_ptr,
                                    length=length,
                                    src_region_offset=src_region_offset,
                                    dst_region_offset=dst_region_offset,
                                    local_block_ids=group_local_block_id,
                                    remote_block_ids=group_remote_block_id,
                                    block_ordinals=block_ordinals,
                                )
                            )
                        src_ptrs.append(src_ptr)
                        dst_ptrs.append(dst_ptr)
                        lengths.append(length)
                    else:
                        for (
                            block_offset,
                            (local_block_id, remote_block_id),
                        ) in enumerate(
                            zip(group_local_block_id, group_remote_block_id),
                        ):
                            src_ptr = (
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptr = (
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            if self.xfer_debug_config is not None:
                                debug_descriptors.append(
                                    self._make_xfer_debug_descriptor(
                                        transfer_id=send_meta.transfer_id,
                                        d_req_id=d_req_id,
                                        descriptor_idx=len(src_ptrs),
                                        region_idx=region_idx,
                                        local_region=local_region,
                                        remote_region=remote_region,
                                        src_ptr=src_ptr,
                                        dst_ptr=dst_ptr,
                                        length=transfer_len,
                                        src_region_offset=src_region_offset,
                                        dst_region_offset=dst_region_offset,
                                        local_block_ids=[local_block_id],
                                        remote_block_ids=[remote_block_id],
                                        block_ordinals=[block_ordinals[block_offset]],
                                    )
                                )
                            src_ptrs.append(src_ptr)
                            dst_ptrs.append(dst_ptr)
                            lengths.append(transfer_len)

                if local_region is local_regions[0]:
                    logger.debug(
                        "Mooncake transfer plan for request %s: local_tp=%d "
                        "remote_tp=%d remote_tp_rank=%d local_block_len=%d "
                        "remote_block_len=%d src_offset=%d dst_offset=%d "
                        "transfer_len=%d coalesce=%s",
                        d_req_id,
                        self.tp_size,
                        agent_meta.remote_tp_size,
                        agent_meta.remote_tp_rank,
                        local_region.block_len,
                        remote_region.block_len,
                        src_region_offset,
                        dst_region_offset,
                        transfer_len,
                        can_coalesce,
                    )

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                len(local_block_ids),
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg, debug_descriptors

    def _bind_sender_thread_device(self) -> None:
        """ThreadPoolExecutor initializer — binds each pool thread to the
        correct CUDA device.  CUDA device selection is thread-local, so
        without this, NVLink transfers fail for TP ranks > 0."""
        current_platform.set_device(self.device_id)

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        duration = time.perf_counter() - start_time
        if ret_value == 0:
            logger.debug(
                "Sending to %s done, took %s",
                remote_session,
                time.perf_counter() - start_time,
            )
        else:
            self.xfer_stats.record_failed_transfer()
            logger.warning(
                "Sending to %s failed (ret=%s) after %s (%d descriptors, %d bytes)",
                remote_session,
                ret_value,
                duration,
                len(src_ptrs),
                sum(lengths),
            )
            logger.warning(
                "Mooncake failed transfer descriptor sample: %s",
                _format_transfer_descriptor_sample(src_ptrs, dst_ptrs, lengths),
            )
        return ret_value

    def _get_shadow_buckets_config(self) -> Any | None:
        assert self.vllm_config.kv_transfer_config is not None
        return self.vllm_config.kv_transfer_config.kv_connector_extra_config.get(
            "shadow_buckets"
        )

    def _should_use_shadow_bucket_metadata(self) -> bool:
        return self.is_kv_producer and self._get_shadow_buckets_config() is not None

    def _build_kv_cache_group_metadata(
        self,
        source_metadata_by_name: dict[str, list[KVCacheAddressMetadata]],
    ) -> None:
        if self.kv_cache_config is None:
            self.kv_cache_group_layer_names = []
            self.kv_cache_group_block_lens = []
            self.kv_cache_layer_materialized_block_lens = {}
            self.layer_to_group_index = {}
            return

        layer_materialized_block_lens: dict[str, int] = {}
        for layer_name, metadata in source_metadata_by_name.items():
            if not metadata:
                continue
            layer_materialized_block_lens[canonical_cache_name(layer_name)] = max(
                item.materialized_block_len or item.block_len for item in metadata
            )

        group_layer_names: list[list[str]] = []
        group_block_lens: list[int] = []
        layer_to_group_index: dict[str, int] = {}
        for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
            group_layer_names.append(list(group.layer_names))
            group_block_len = 0
            for layer_name in group.layer_names:
                layer_to_group_index[canonical_cache_name(layer_name)] = group_idx
                metadata = source_metadata_by_name.get(layer_name)
                if metadata and group_block_len == 0:
                    group_block_len = max(
                        item.materialized_block_len or item.block_len
                        for item in metadata
                    )
            if group_block_len == 0:
                group_block_len = int(group.kv_cache_spec.page_size_bytes)
            group_block_lens.append(group_block_len)

        self.kv_cache_group_layer_names = group_layer_names
        self.kv_cache_group_block_lens = group_block_lens
        self.kv_cache_layer_materialized_block_lens = layer_materialized_block_lens
        self.layer_to_group_index = layer_to_group_index

    def _build_shadow_bucket_metadata(
        self,
        source_metadata_by_name: dict[str, list[KVCacheAddressMetadata]],
    ) -> tuple[list[int], list[int], list[int], tuple[ShadowPlacement, ...]]:
        if self.kv_cache_config is None:
            raise RuntimeError(
                "Mooncake shadow bucket metadata requires KVCacheConfig."
            )

        producer_buckets = bucket_slots_from_kv_cache_tensors(
            self.kv_cache_config.kv_cache_tensors, self.num_blocks
        )
        producer_bucket_layer_names = {
            (bucket.page_size, bucket.slot_idx): bucket.layer_names
            for bucket in producer_buckets
        }
        consumer_buckets = parse_shadow_buckets(self._get_shadow_buckets_config())
        placements = build_shadow_bucket_plan(producer_buckets, consumer_buckets)

        shadow_base_addrs: list[int] = []
        source_block_lens: list[int] = []
        target_block_lens: list[int] = []
        shadow_sources_by_layer: dict[str, ShadowTransferSource] = {}
        for placement in placements:
            if len(placement.source_bucket_keys) > 1:
                logger.warning(
                    "Mooncake shadow target bucket page_size=%d slot_idx=%d spans "
                    "multiple producer buckets %s. Metadata-only shadow transfer "
                    "will use producer bucket %s; a real shadow buffer/repack may "
                    "be required for correct heterogeneous MLA cache contents.",
                    placement.target_page_size,
                    placement.target_slot_idx,
                    placement.source_bucket_keys,
                    (placement.source.page_size, placement.source.slot_idx),
                )
            for layer_mapping in placement.layer_mappings:
                source_metadata = source_metadata_by_name.get(
                    layer_mapping.source.layer_name, ()
                )
                if not source_metadata:
                    raise RuntimeError(
                        "Mooncake shadow bucket source cache is absent from "
                        f"kv_caches: {layer_mapping.source.layer_name!r}."
                    )
                group_idx = self.layer_to_group_index.get(
                    canonical_cache_name(layer_mapping.source.layer_name)
                )
                if group_idx is None:
                    raise RuntimeError(
                        "Mooncake shadow bucket source cache is absent from "
                        "kv_cache_groups: "
                        f"{layer_mapping.source.layer_name!r}."
                    )
                source_base_addr = min(
                    metadata.base_addr for metadata in source_metadata
                )
                source_block_len = max(
                    metadata.block_len for metadata in source_metadata
                )
                source_materialized_len = max(
                    metadata.materialized_block_len or metadata.block_len
                    for metadata in source_metadata
                )
                shadow_sources_by_layer[
                    canonical_cache_name(layer_mapping.source.layer_name)
                ] = ShadowTransferSource(
                    layer_name=layer_mapping.source.layer_name,
                    base_addr=source_base_addr,
                    block_len=source_block_len,
                    kv_block_len=source_block_len,
                    materialized_block_len=source_materialized_len,
                    group_idx=group_idx,
                )
            source_bucket_key = (placement.source.page_size, placement.source.slot_idx)
            source_layer_names = producer_bucket_layer_names[source_bucket_key]
            source_metadata = [
                metadata
                for layer_name in source_layer_names
                for metadata in source_metadata_by_name.get(layer_name, ())
            ]
            if not source_metadata:
                raise RuntimeError(
                    "Mooncake shadow bucket source cache is absent from kv_caches: "
                    f"{placement.source.layer_name!r}."
                )
            shadow_base_addrs.append(
                min(metadata.base_addr for metadata in source_metadata)
            )
            source_block_lens.append(placement.source.page_size)
            target_block_lens.append(placement.target_page_size)

        self.shadow_sources_by_layer = shadow_sources_by_layer
        logger.info(
            "Enabled Mooncake shadow bucket metadata with %d placements. "
            "source_block_lens=%s target_block_lens=%s",
            len(placements),
            source_block_lens,
            target_block_lens,
        )
        return shadow_base_addrs, source_block_lens, target_block_lens, placements

    def _get_shadow_transfer_regions(self) -> list[TransferRegion]:
        return self._get_transfer_regions(
            self.kv_caches_base_addr, self.block_len_per_layer
        )

    def _get_shadow_target_transfer_regions(self) -> list[TransferRegion]:
        return self._get_transfer_regions(
            [0] * len(self.shadow_block_len_per_layer),
            self.shadow_block_len_per_layer,
        )

    def _get_shadow_copy_len(
        self,
        local_region: TransferRegion,
        remote_region: TransferRegion,
        src_region_offset: int,
        dst_region_offset: int,
        transfer_len: int,
    ) -> int:
        src_available = local_region.kv_block_len - src_region_offset
        dst_available = remote_region.kv_block_len - dst_region_offset
        return min(transfer_len, src_available, dst_available)

    def _validate_shadow_region_expansion(
        self,
        source_regions: list[TransferRegion],
    ) -> str | None:
        target_regions = self._get_transfer_regions(
            [0] * len(self.shadow_block_len_per_layer),
            self.shadow_block_len_per_layer,
        )
        if len(source_regions) != len(target_regions):
            return "Mooncake shadow transfer region expansion changed region count."
        return None

    def _validate_shadow_transfer_regions(
        self,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> str | None:
        if len(local_regions) != len(remote_regions):
            return (
                "Mooncake shadow bucket requires matching KV region counts between "
                "producer shadow metadata and consumer metadata."
            )
        expansion_err = self._validate_shadow_region_expansion(local_regions)
        if expansion_err is not None:
            return expansion_err
        for idx, (local_region, remote_region) in enumerate(
            zip(local_regions, remote_regions)
        ):
            if local_region.kv_block_len != remote_region.kv_block_len:
                logger.warning_once(
                    "Mooncake shadow bucket will transfer only %d bytes between "
                    "local region %d with %d bytes per block and remote with "
                    "%d bytes per block.",
                    min(local_region.kv_block_len, remote_region.kv_block_len),
                    idx,
                    local_region.kv_block_len,
                    remote_region.kv_block_len,
                )
        return None

    @staticmethod
    def _as_cache_region_list(
        cache_or_caches: KVCacheValue,
        split_k_and_v: bool,
    ) -> list[torch.Tensor]:
        if isinstance(cache_or_caches, torch.Tensor):
            return list(cache_or_caches) if split_k_and_v else [cache_or_caches]

        if isinstance(cache_or_caches, list | tuple):
            return list(cache_or_caches)

        raise TypeError(
            "MooncakeConnector expected a KV cache tensor or a sequence of "
            f"tensors, got {type(cache_or_caches).__name__}."
        )

    def register_kv_caches(self, kv_caches: dict[str, KVCacheValue]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        register_ranges_by_storage: dict[int, list[tuple[int, int]]] = {}
        seen_base_addresses = []
        self.block_len_per_layer = []
        self._xfer_debug_cache_views = []
        source_metadata_by_name: dict[str, list[KVCacheAddressMetadata]] = {}

        split_k_and_v = self.transfer_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = self._as_cache_region_list(cache_or_caches, split_k_and_v)
            source_metadata: list[KVCacheAddressMetadata] = []
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()

                if tensor_size_bytes is None:
                    tensor_size_bytes = cache.nbytes
                    self.num_blocks = cache.shape[0]
                assert (
                    cache.shape[0] == self.num_blocks
                ), "All kv cache tensors must have the same number of blocks"

                # Use stride-based block length so RDMA reaches the last
                # block's padding (e.g. DeepseekV4 MLA alignment). stride(0)
                # reflects the actual byte distance between consecutive
                # blocks in GPU memory, which matches or exceeds the
                # shape-based size.
                block_len = cache.stride(0) * cache.element_size()
                if self.xfer_debug_config is not None:
                    self._xfer_debug_cache_views.append(
                        XferDebugCacheView(
                            layer_name=layer_name,
                            tensor=cache,
                            base_addr=base_addr,
                            block_len=block_len,
                            materialized_block_len=(
                                cache[0].numel() * cache.element_size()
                            ),
                        )
                    )

                source_metadata.append(
                    KVCacheAddressMetadata(
                        base_addr=base_addr,
                        block_len=block_len,
                        materialized_block_len=(
                            cache[0].numel() * cache.element_size()
                        ),
                    )
                )
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)

                self.block_len_per_layer.append(block_len)
                register_len = self.num_blocks * block_len
                storage_key = _get_tensor_storage_key(cache)
                register_ranges_by_storage.setdefault(storage_key, []).append(
                    (base_addr, base_addr + register_len)
                )

            source_metadata_by_name[layer_name] = source_metadata

        self._build_kv_cache_group_metadata(source_metadata_by_name)

        if self._should_use_shadow_bucket_metadata():
            (
                self.kv_caches_base_addr,
                self.block_len_per_layer,
                self.shadow_block_len_per_layer,
                self.shadow_bucket_plan,
            ) = self._build_shadow_bucket_metadata(source_metadata_by_name)
        else:
            self.kv_caches_base_addr = seen_base_addresses
            self.shadow_block_len_per_layer = []
            self.shadow_bucket_plan = ()
            self.shadow_sources_by_layer = {}

        self.seen_base_addresses = seen_base_addresses

        register_ptrs, register_lens = _merge_register_ranges(
            register_ranges_by_storage
        )
        ret_value = self.engine.batch_register_memory(register_ptrs, register_lens)
        if ret_value != 0:
            raise RuntimeError(
                "Mooncake batch memory registration failed "
                f"with ret_value={ret_value}, num_regions={len(register_ptrs)}, "
                f"lengths={register_lens}."
            )

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_lens=%s",
            self.num_blocks,
            self.block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                self.xfer_stats.record_kv_expired_req()
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(self) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        recv_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_reqs(), self.receiver_loop
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        finished_recving_reqs = recv_fut.result() if recv_fut else set()
        finished_sending_reqs = send_fut.result() if send_fut else set()

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        """Return transfer stats collected since the last call, or None
        if nothing has been recorded in this interval."""
        if self.xfer_stats.is_empty():
            return None
        return self.xfer_stats.clone_and_reset()

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            kv_cache_group_layer_names=self.kv_cache_group_layer_names,
            kv_cache_group_block_lens=self.kv_cache_group_block_lens,
            kv_cache_layer_materialized_block_lens=(
                self.kv_cache_layer_materialized_block_lens
            ),
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        self.xfer_stats.record_failed_recv()
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            self.xfer_stats.record_failed_recv()
            return

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []
        self._dump_receiver_xfer_debug(response, set(ok_reqs))

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if (
                pull_meta.pull_tasks_count == 0
                and any(pull_meta.local_block_ids)
            ):
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    def _dump_receiver_xfer_debug(
        self,
        response: MooncakeXferResponse,
        ok_req_ids: set[ReqId],
    ) -> None:
        if self.xfer_debug_config is None or not response.debug_descriptors:
            return
        descriptors: list[XferDebugDescriptor] = []
        for raw_descriptor in response.debug_descriptors:
            try:
                descriptor = descriptor_from_dict(raw_descriptor)
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Invalid Mooncake xfer debug descriptor: %s", exc)
                continue
            if descriptor.d_req_id in ok_req_ids:
                descriptors.append(descriptor)
        self._dump_xfer_debug_records(
            side="consumer",
            descriptors=descriptors,
            pointer_kind="destination",
        )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict[str, Any] = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.transfer_topo.handshake_target_ranks(
            self._tp_size[remote_engine_id]
        )
        count = len(remote_tp_ranks)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            if block_ids:
                # Already gone through request_finished()
                send_meta = self.reqs_need_send[transfer_id]
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if transfer_id not in self.reqs_need_send:
                    self.reqs_need_send[transfer_id] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                    )
        for transfer_id in metadata.reqs_not_processed:
            send_meta = self.reqs_need_send.pop(transfer_id)
            if send_meta:
                assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    def _producer_cache_is_replicated(self) -> bool:
        return self.transfer_topo.local_replicates_kv_cache

    def _get_transfer_regions(
        self, base_addrs: list[int], block_lens: list[int]
    ) -> list[TransferRegion]:
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            is_kv_layout_blocks_first=self.transfer_topo.virtually_split_kv_in_blocks,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
