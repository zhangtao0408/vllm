# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    KVCacheAddressMetadata,
    MooncakeConnectorWorker,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor


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
            KVCacheAddressMetadata(base_addr=1200, block_len=100),
            KVCacheAddressMetadata(base_addr=1000, block_len=200),
            KVCacheAddressMetadata(base_addr=1400, block_len=50),
        ],
        "model.layers.2.self_attn.indexer.compressor.state_cache": [
            KVCacheAddressMetadata(base_addr=1100, block_len=80),
        ],
        "model.layers.2.self_attn.attn": [
            KVCacheAddressMetadata(base_addr=5200, block_len=300),
        ],
        "model.layers.2.self_attn.compressor.state_cache": [
            KVCacheAddressMetadata(base_addr=5000, block_len=400),
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
