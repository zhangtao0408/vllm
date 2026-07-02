# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVCacheAddressMetadata,
    MooncakeConnectorWorker,
    MooncakeXferMetadata,
    SendBlockMeta,
    ShadowTransferSource,
    TransferRegion,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.shadow_bucket import (
    ShadowLayerMapping,
    ShadowPlacement,
    ShadowSource,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.xfer_debug import (
    XferDebugConfig,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


class ShadowTransferTestWorker(MooncakeConnectorWorker):
    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return True, 0, 0, remote_kv_block_len


def test_shadow_bucket_metadata_uses_physical_bucket_for_tuple_cache():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.num_blocks = 10
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=10,
        kv_cache_tensors=[
            KVCacheTensor(
                size=8448 * 10,
                shared_by=[
                    "model.layers.2.self_attn.indexer.k_cache",
                    "model.layers.2.self_attn.indexer.compressor.state_cache",
                ],
            ),
            KVCacheTensor(
                size=40960 * 10,
                shared_by=[
                    "model.layers.2.self_attn.attn",
                    "model.layers.2.self_attn.compressor.state_cache",
                ],
            ),
        ],
        kv_cache_groups=[],
    )
    worker.layer_to_group_index = {
        "model.layers.2.attn.indexer.k_cache": 0,
        "model.layers.2.attn.indexer.compressor.state_cache": 1,
        "model.layers.2.attn": 2,
        "model.layers.2.attn.compressor.state_cache": 3,
    }
    worker._get_shadow_buckets_config = lambda: [
        {
            "page_size": 8640,
            "layer_names": [
                "model.layers.2.attn.indexer.k_cache",
                "model.layers.2.attn.indexer.compressor.state_cache",
            ],
        },
        {
            "page_size": 37440,
            "layer_names": [
                "model.layers.2.attn",
                "model.layers.2.attn.compressor.state_cache",
            ],
        },
    ]
    source_metadata_by_name = {
        "model.layers.2.self_attn.indexer.k_cache": [
            KVCacheAddressMetadata(
                base_addr=1200, block_len=100, materialized_block_len=90
            ),
            KVCacheAddressMetadata(
                base_addr=1000, block_len=200, materialized_block_len=180
            ),
            KVCacheAddressMetadata(
                base_addr=1400, block_len=50, materialized_block_len=40
            ),
        ],
        "model.layers.2.self_attn.indexer.compressor.state_cache": [
            KVCacheAddressMetadata(
                base_addr=1100, block_len=80, materialized_block_len=70
            ),
        ],
        "model.layers.2.self_attn.attn": [
            KVCacheAddressMetadata(
                base_addr=5200, block_len=300, materialized_block_len=280
            ),
        ],
        "model.layers.2.self_attn.compressor.state_cache": [
            KVCacheAddressMetadata(
                base_addr=5000, block_len=400, materialized_block_len=320
            ),
        ],
    }

    base_addrs, source_block_lens, target_block_lens, placements = (
        worker._build_shadow_bucket_metadata(source_metadata_by_name)
    )

    assert base_addrs == [1000, 5000]
    assert source_block_lens == [8448, 40960]
    assert target_block_lens == [8640, 37440]
    assert placements[0].source.layer_name == (
        "model.layers.2.self_attn.indexer.k_cache"
    )
    assert (
        worker.shadow_sources_by_layer["model.layers.2.attn.indexer.k_cache"].base_addr
        == 1000
    )
    assert (
        worker.shadow_sources_by_layer[
            "model.layers.2.attn.indexer.k_cache"
        ].materialized_block_len
        == 180
    )
    assert (
        worker.shadow_sources_by_layer[
            "model.layers.2.attn.compressor.state_cache"
        ].group_idx
        == 3
    )
    assert worker.kv_cache_layer_materialized_block_lens == {
        "model.layers.2.attn.indexer.k_cache": 180,
        "model.layers.2.attn.indexer.compressor.state_cache": 70,
        "model.layers.2.attn": 280,
        "model.layers.2.attn.compressor.state_cache": 320,
    }


def test_shadow_transfer_uses_logical_group_blocks(tmp_path):
    worker = object.__new__(ShadowTransferTestWorker)
    worker.tp_rank = 0
    worker.xfer_debug_config = XferDebugConfig(dump_dir=tmp_path)
    worker.shadow_bucket_plan = (
        ShadowPlacement(
            target_page_size=37440,
            target_slot_idx=0,
            target_layer_names=(
                "model.layers.2.attn",
                "model.layers.0.attn.swa_cache",
            ),
            source=ShadowSource(
                page_size=40960,
                slot_idx=0,
                layer_name="model.layers.2.self_attn.attn",
            ),
            source_bucket_keys=((40960, 0),),
            layer_mappings=(
                ShadowLayerMapping(
                    target_layer_name="model.layers.2.attn",
                    source=ShadowSource(
                        page_size=40960,
                        slot_idx=0,
                        layer_name="model.layers.2.self_attn.attn",
                    ),
                ),
                ShadowLayerMapping(
                    target_layer_name="model.layers.0.attn.swa_cache",
                    source=ShadowSource(
                        page_size=40960,
                        slot_idx=0,
                        layer_name="model.layers.0.self_attn.swa_cache",
                    ),
                ),
            ),
        ),
    )
    worker.shadow_sources_by_layer = {
        "model.layers.2.attn": ShadowTransferSource(
            layer_name="model.layers.2.self_attn.attn",
            base_addr=0x100000,
            block_len=40960,
            kv_block_len=40960,
            materialized_block_len=40960,
            group_idx=0,
        ),
        "model.layers.0.attn.swa_cache": ShadowTransferSource(
            layer_name="model.layers.0.self_attn.swa_cache",
            base_addr=0x200000,
            block_len=40960,
            kv_block_len=40960,
            materialized_block_len=40960,
            group_idx=1,
        ),
    }
    send_meta = SendBlockMeta(
        p_req_id="p",
        transfer_id="xfer",
        local_block_ids=[[10], [20]],
        ready=asyncio.Event(),
    )
    agent_meta = MooncakeXferMetadata(
        remote_hostname="h20",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d": ("xfer", [[30], [40]])},
        kv_caches_base_addr=[0x300000],
        block_lens=[37440],
        kv_cache_group_layer_names=[
            ["model.layers.2.attn"],
            ["model.layers.0.attn.swa_cache"],
        ],
        kv_cache_group_block_lens=[37376, 37376],
    )
    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    lengths: list[int] = []
    debug_descriptors = []

    err = worker._append_shadow_transfer_params(
        d_req_id="d",
        send_meta=send_meta,
        remote_block_ids_per_group=agent_meta.req_blocks["d"][1],
        agent_meta=agent_meta,
        remote_regions=[
            TransferRegion(
                base_addr=0x300000,
                block_len=37440,
                kv_block_len=37440,
            )
        ],
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        lengths=lengths,
        debug_descriptors=debug_descriptors,
    )

    assert err is None
    assert len(src_ptrs) == 128
    assert lengths == [584] * 128
    assert src_ptrs[0] == 0x100000 + 10 * 40960
    assert dst_ptrs[0] == 0x300000 + 30 * 37440
    assert debug_descriptors[0].source_layer == "model.layers.2.self_attn.attn"
    assert debug_descriptors[0].target_layers == ("model.layers.2.attn",)
    assert src_ptrs[64] == 0x200000 + 20 * 40960
    assert dst_ptrs[64] == 0x300000 + 40 * 37440
    assert debug_descriptors[64].source_layer == ("model.layers.0.self_attn.swa_cache")
    assert debug_descriptors[64].target_layers == ("model.layers.0.attn.swa_cache",)


def test_shadow_transfer_uses_target_layer_materialized_len(tmp_path):
    worker = object.__new__(ShadowTransferTestWorker)
    worker.tp_rank = 0
    worker.xfer_debug_config = XferDebugConfig(dump_dir=tmp_path)
    worker.shadow_bucket_plan = (
        ShadowPlacement(
            target_page_size=37440,
            target_slot_idx=0,
            target_layer_names=("model.layers.2.attn",),
            source=ShadowSource(
                page_size=40960,
                slot_idx=0,
                layer_name="model.layers.2.self_attn.attn",
            ),
            source_bucket_keys=((40960, 0),),
            layer_mappings=(
                ShadowLayerMapping(
                    target_layer_name="model.layers.2.attn",
                    source=ShadowSource(
                        page_size=40960,
                        slot_idx=0,
                        layer_name="model.layers.2.self_attn.attn",
                    ),
                ),
            ),
        ),
    )
    worker.shadow_sources_by_layer = {
        "model.layers.2.attn": ShadowTransferSource(
            layer_name="model.layers.2.self_attn.attn",
            base_addr=0x100000,
            block_len=40960,
            kv_block_len=40960,
            materialized_block_len=40960,
            group_idx=0,
        ),
    }
    send_meta = SendBlockMeta(
        p_req_id="p",
        transfer_id="xfer",
        local_block_ids=[[10]],
        ready=asyncio.Event(),
    )
    agent_meta = MooncakeXferMetadata(
        remote_hostname="h20",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"d": ("xfer", [[30]])},
        kv_caches_base_addr=[0x300000],
        block_lens=[37440],
        kv_cache_group_layer_names=[["model.layers.2.attn"]],
        kv_cache_group_block_lens=[8448],
        kv_cache_layer_materialized_block_lens={
            "model.layers.2.attn": 37376,
        },
    )
    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    lengths: list[int] = []
    debug_descriptors = []

    err = worker._append_shadow_transfer_params(
        d_req_id="d",
        send_meta=send_meta,
        remote_block_ids_per_group=agent_meta.req_blocks["d"][1],
        agent_meta=agent_meta,
        remote_regions=[
            TransferRegion(
                base_addr=0x300000,
                block_len=37440,
                kv_block_len=37440,
            )
        ],
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        lengths=lengths,
        debug_descriptors=debug_descriptors,
    )

    assert err is None
    assert len(src_ptrs) == 64
    assert lengths == [584] * 64
    assert debug_descriptors[0].target_block_len == 37376
