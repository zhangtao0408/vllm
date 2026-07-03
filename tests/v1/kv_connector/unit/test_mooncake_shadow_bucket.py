# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.shadow_bucket import (
    BucketSlot,
    build_shadow_bucket_plan,
)


def test_shadow_bucket_plan_keeps_first_logical_source_alias():
    producer_buckets = (
        BucketSlot(
            page_size=8448,
            slot_idx=0,
            layer_names=(
                "model.layers.2.self_attn.indexer.k_cache",
                "model.layers.2.self_attn.indexer.compressor.state_cache",
            ),
        ),
        BucketSlot(
            page_size=40960,
            slot_idx=0,
            layer_names=(
                "model.layers.2.self_attn.attn",
                "model.layers.0.self_attn.swa_cache",
                "model.layers.1.self_attn.swa_cache",
                "model.layers.2.self_attn.compressor.state_cache",
                "model.layers.3.self_attn.compressor.state_cache",
            ),
        ),
    )
    consumer_buckets = (
        BucketSlot(
            page_size=8640,
            slot_idx=0,
            layer_names=(
                "model.layers.2.attn.indexer.k_cache",
                "model.layers.2.attn.indexer.compressor.state_cache",
            ),
        ),
        BucketSlot(
            page_size=37440,
            slot_idx=0,
            layer_names=(
                "model.layers.2.attn",
                "model.layers.0.attn.swa_cache",
                "model.layers.1.attn.swa_cache",
                "model.layers.2.attn.compressor.state_cache",
                "model.layers.3.attn.compressor.state_cache",
            ),
        ),
    )

    placements = build_shadow_bucket_plan(producer_buckets, consumer_buckets)

    assert placements[0].source.layer_name == (
        "model.layers.2.self_attn.indexer.k_cache"
    )
    assert placements[1].source.layer_name == "model.layers.2.self_attn.attn"
    assert [
        (mapping.target_layer_name, mapping.source.layer_name)
        for mapping in placements[1].layer_mappings
    ] == [
        ("model.layers.2.attn", "model.layers.2.self_attn.attn"),
        (
            "model.layers.0.attn.swa_cache",
            "model.layers.0.self_attn.swa_cache",
        ),
        (
            "model.layers.1.attn.swa_cache",
            "model.layers.1.self_attn.swa_cache",
        ),
        (
            "model.layers.2.attn.compressor.state_cache",
            "model.layers.2.self_attn.compressor.state_cache",
        ),
        (
            "model.layers.3.attn.compressor.state_cache",
            "model.layers.3.self_attn.compressor.state_cache",
        ),
    ]
