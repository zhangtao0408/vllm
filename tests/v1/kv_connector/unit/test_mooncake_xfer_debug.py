# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import torch

from examples.disaggregated.mooncake_connector.compare_xfer_debug import (
    compare_decode_golden,
    compare_prefill_decode,
    load_records,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.xfer_debug import (
    XferDebugCacheView,
    XferDebugConfig,
    XferDebugDescriptor,
    XferDebugDumpRequest,
    descriptor_from_dict,
    descriptor_to_dict,
    dump_xfer_debug_records,
    parse_xfer_debug_config,
)
from vllm.v1.worker.kv_xfer_debug import (
    NativeKVCacheDescriptorRequest,
    build_native_kv_cache_descriptors,
    build_xfer_debug_cache_views,
)


def test_parse_xfer_debug_config_from_json_path(tmp_path: Path):
    config_path = tmp_path / "xfer_debug.json"
    config_path.write_text(
        json.dumps(
            {
                "dump_dir": "~/tmp/pd_xfer_4l/h20_golden",
                "max_requests": 2,
                "dump_on_prefill": True,
            }
        )
    )

    config = parse_xfer_debug_config(str(config_path))

    assert config is not None
    assert config.dump_dir.name == "h20_golden"
    assert config.max_requests == 2
    assert config.dump_on_prefill is True


def test_xfer_debug_dump_records_source_slice_and_full_block(tmp_path: Path):
    tensor = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
    descriptor = XferDebugDescriptor(
        transfer_id="transfer-1",
        d_req_id="d-req-1",
        tp_rank=0,
        descriptor_idx=0,
        region_idx=2,
        block_id=1,
        local_block_id=1,
        remote_block_id=7,
        local_block_ids=(1,),
        remote_block_ids=(7,),
        block_ordinals=(0,),
        src_ptr=tensor.data_ptr() + 16 + 4,
        dst_ptr=0,
        length=8,
        src_offset=4,
        dst_offset=0,
        source_layer="model.layers.3.attn",
        target_layers=("model.layers.3.attn",),
        source_block_len=16,
        target_block_len=8,
        bucket_type="c128_1728",
    )

    dumped = dump_xfer_debug_records(
        XferDebugDumpRequest(
            config=XferDebugConfig(
                dump_dir=tmp_path,
                max_requests=1,
                dump_full_source_block=True,
            ),
            cache_views=(
                XferDebugCacheView(
                    layer_name="model.layers.3.attn",
                    tensor=tensor,
                    base_addr=tensor.data_ptr(),
                    block_len=16,
                ),
            ),
            side="producer",
            pointer_kind="source",
            descriptors=(descriptor,),
        )
    )

    records = load_records(tmp_path)
    assert dumped == 1
    assert len(records) == 1
    assert records[0]["payload"].tolist() == list(range(20, 28))
    assert records[0]["full_source_block"].tolist() == list(range(16, 32))
    assert records[0]["descriptor"]["bucket_type"] == "c128_1728"
    assert records[0]["tensor_name"] == "model.layers.3.attn"
    assert "tensor_model.layers.3.attn" in Path(records[0]["path"]).name
    assert descriptor_from_dict(descriptor_to_dict(descriptor)) == descriptor


def test_native_prefill_descriptors_follow_kv_cache_tensor_order():
    cache_a = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
    cache_b = torch.arange(128, dtype=torch.uint8).reshape(4, 32)
    kv_cache_config = _FakeKVCacheConfig(
        kv_cache_tensors=(
            _FakeKVCacheTensor(shared_by=("layer.a", "layer.b")),
            _FakeKVCacheTensor(shared_by=("layer.c",)),
        ),
        kv_cache_groups=(
            _FakeKVCacheGroup(layer_names=("layer.a",)),
            _FakeKVCacheGroup(layer_names=("layer.b",)),
            _FakeKVCacheGroup(layer_names=("layer.c",)),
        ),
    )

    group_block_ids = ((3, 1), (2,), (0,))
    descriptors = build_native_kv_cache_descriptors(
        NativeKVCacheDescriptorRequest(
            kv_cache_config=kv_cache_config,
            kv_caches={"layer.a": cache_a, "layer.b": cache_a, "layer.c": cache_b},
            request_id="req-1",
            transfer_id="native-prefill",
            tp_rank=0,
            group_block_ids=group_block_ids,
        )
    )

    assert group_block_ids == ((3, 1), (2,), (0,))
    assert [(d.region_idx, d.block_id, d.length) for d in descriptors] == [
        (0, 3, 16),
        (0, 1, 16),
        (0, 2, 16),
        (1, 0, 32),
    ]
    assert descriptors[1].block_ordinals == (1,)
    assert descriptors[2].source_layer == "layer.b"
    assert descriptors[2].block_ordinals == (2,)
    assert descriptors[3].target_layers == ("layer.c",)
    assert [
        (
            view.layer_name,
            view.base_addr,
            view.block_len,
            view.materialized_block_len,
        )
        for view in build_xfer_debug_cache_views({"layer.a": cache_a})
    ] == [("layer.a", cache_a.data_ptr(), 16, 16)]


def test_native_prefill_dump_uses_materialized_block_len_for_padded_stride(
    tmp_path: Path,
):
    storage = torch.arange(72, dtype=torch.uint8)
    padded_cache = torch.as_strided(storage, size=(4, 16), stride=(18, 1))
    kv_cache_config = _FakeKVCacheConfig(
        kv_cache_tensors=(_FakeKVCacheTensor(shared_by=("layer.padded",)),),
        kv_cache_groups=(_FakeKVCacheGroup(layer_names=("layer.padded",)),),
    )

    descriptors = build_native_kv_cache_descriptors(
        NativeKVCacheDescriptorRequest(
            kv_cache_config=kv_cache_config,
            kv_caches={"layer.padded": padded_cache},
            request_id="req-1",
            transfer_id="native-prefill",
            tp_rank=0,
            group_block_ids=((2,),),
        )
    )
    dumped = dump_xfer_debug_records(
        XferDebugDumpRequest(
            config=XferDebugConfig(dump_dir=tmp_path),
            cache_views=build_xfer_debug_cache_views({"layer.padded": padded_cache}),
            side="golden",
            pointer_kind="source",
            descriptors=descriptors,
            rank_tag="dp3__tp0",
        )
    )

    records = load_records(tmp_path)
    assert dumped == 1
    assert len(records) == 1
    assert descriptors[0].length == 16
    assert descriptors[0].source_block_len == 16
    assert descriptors[0].target_block_len == 18
    assert records[0]["payload"].tolist() == list(range(36, 52))
    assert records[0]["block_stride_bytes"] == 18
    assert records[0]["materialized_block_bytes"] == 16
    assert records[0]["rank_tag"] == "dp3__tp0"
    record_name = Path(records[0]["path"]).name
    assert record_name.startswith("dp3__tp0__golden__")
    assert "tensor_layer.padded" in record_name


def test_xfer_debug_dump_preserves_padded_descriptor_len(tmp_path: Path):
    storage = torch.arange(72, dtype=torch.uint8)
    padded_cache = torch.as_strided(storage, size=(4, 16), stride=(18, 1))
    descriptor = XferDebugDescriptor(
        transfer_id="transfer-1",
        d_req_id="d-req-1",
        tp_rank=0,
        descriptor_idx=0,
        region_idx=0,
        block_id=2,
        local_block_id=2,
        remote_block_id=2,
        local_block_ids=(2,),
        remote_block_ids=(2,),
        block_ordinals=(0,),
        src_ptr=padded_cache.data_ptr() + 2 * 18,
        dst_ptr=0,
        length=18,
        src_offset=0,
        dst_offset=0,
        source_layer="layer.padded",
        target_layers=("layer.padded",),
        source_block_len=18,
        target_block_len=18,
        bucket_type="18",
    )

    dump_xfer_debug_records(
        XferDebugDumpRequest(
            config=XferDebugConfig(dump_dir=tmp_path),
            cache_views=build_xfer_debug_cache_views(
                {"layer.padded": padded_cache}
            ),
            side="producer",
            pointer_kind="source",
            descriptors=(descriptor,),
        )
    )

    records = load_records(tmp_path)
    assert len(records) == 1
    assert records[0]["payload"].tolist() == list(range(36, 54))
    assert records[0]["payload_num_bytes"] == 18
    assert records[0]["payload_materialized_num_bytes"] == 16
    assert records[0]["payload_padding_num_bytes"] == 2
    assert records[0]["block_stride_bytes"] == 18
    assert records[0]["materialized_block_bytes"] == 16


def test_xfer_debug_prefers_descriptor_layer_name_for_shared_tensor(
    tmp_path: Path,
):
    cache = torch.arange(64, dtype=torch.uint8).reshape(4, 16)
    descriptor = XferDebugDescriptor(
        transfer_id="transfer-1",
        d_req_id="d-req-1",
        tp_rank=0,
        descriptor_idx=0,
        region_idx=0,
        block_id=1,
        local_block_id=1,
        remote_block_id=1,
        local_block_ids=(1,),
        remote_block_ids=(1,),
        block_ordinals=(0,),
        src_ptr=cache.data_ptr() + 16,
        dst_ptr=cache.data_ptr() + 16,
        length=16,
        src_offset=0,
        dst_offset=0,
        source_layer="layer.b",
        target_layers=("layer.a", "layer.b"),
        source_block_len=16,
        target_block_len=16,
        bucket_type="16",
    )

    dump_xfer_debug_records(
        XferDebugDumpRequest(
            config=XferDebugConfig(dump_dir=tmp_path),
            cache_views=build_xfer_debug_cache_views(
                {
                    "layer.a": cache,
                    "layer.b": cache,
                }
            ),
            side="golden",
            pointer_kind="source",
            descriptors=(descriptor,),
        )
    )

    records = load_records(tmp_path)
    assert len(records) == 1
    assert records[0]["tensor_name"] == "layer.a"
    assert records[0]["physical_tensor_name"] == "layer.b"
    assert records[0]["logical_tensor_names"] == ("layer.a", "layer.b")
    assert "tensor_layer.a" in Path(records[0]["path"]).name


def test_compare_xfer_debug_detects_byte_and_golden_mismatch(tmp_path: Path):
    prefill_dir = tmp_path / "prefill"
    decode_dir = tmp_path / "decode"
    golden_dir = tmp_path / "golden"
    prefill_dir.mkdir()
    decode_dir.mkdir()
    golden_dir.mkdir()

    descriptor = _descriptor()
    _save_record(prefill_dir / "p.pt", "producer", descriptor, b"\x01\x02\x03\x04")
    _save_record(decode_dir / "d.pt", "consumer", descriptor, b"\x01\x02\x00\x04")
    _save_record(golden_dir / "g.pt", "golden", descriptor, b"\x01\x02\x03\x04")

    byte_results = compare_prefill_decode(
        load_records(prefill_dir), load_records(decode_dir)
    )
    golden_results = compare_decode_golden(
        load_records(decode_dir), load_records(golden_dir)
    )

    assert len(byte_results) == 1
    assert byte_results[0].equal is False
    assert byte_results[0].first_diff == 2
    assert len(golden_results) == 1
    assert golden_results[0].length_match is True
    assert golden_results[0].max_abs is not None
    assert golden_results[0].max_abs > 0


def test_compare_xfer_debug_matches_payload_when_wire_ids_differ(tmp_path: Path):
    prefill_dir = tmp_path / "prefill"
    decode_dir = tmp_path / "decode"
    prefill_dir.mkdir()
    decode_dir.mkdir()

    prefill_descriptor = _descriptor()
    decode_descriptor = {
        **prefill_descriptor,
        "transfer_id": "decode-local-transfer",
        "d_req_id": "decode-local-request",
        "descriptor_idx": 9,
    }
    _save_record(
        prefill_dir / "p.pt",
        "producer",
        prefill_descriptor,
        b"\x01\x02\x03\x04",
    )
    _save_record(
        decode_dir / "d.pt",
        "consumer",
        decode_descriptor,
        b"\x01\x02\x03\x04",
    )

    byte_results = compare_prefill_decode(
        load_records(prefill_dir), load_records(decode_dir)
    )

    assert len(byte_results) == 1
    assert byte_results[0].equal is True
    assert byte_results[0].key[0] == "payload"


def test_compare_xfer_debug_ignores_padding_for_golden_stats(tmp_path: Path):
    decode_dir = tmp_path / "decode"
    golden_dir = tmp_path / "golden"
    decode_dir.mkdir()
    golden_dir.mkdir()

    descriptor = _descriptor()
    _save_record(
        decode_dir / "d.pt",
        "consumer",
        descriptor,
        b"\x01\x02\x03\x04\xaa\xbb",
        materialized_block_bytes=4,
    )
    _save_record(
        golden_dir / "g.pt",
        "golden",
        descriptor,
        b"\x01\x02\x03\x04\xcc\xdd",
        materialized_block_bytes=4,
    )

    golden_results = compare_decode_golden(
        load_records(decode_dir), load_records(golden_dir)
    )

    assert len(golden_results) == 1
    assert golden_results[0].length_match is True
    assert golden_results[0].compared_bytes == 4
    assert golden_results[0].max_abs == 0.0
    assert golden_results[0].allclose is True


def test_compare_xfer_debug_expands_decode_logical_aliases(tmp_path: Path):
    decode_dir = tmp_path / "decode"
    golden_dir = tmp_path / "golden"
    decode_dir.mkdir()
    golden_dir.mkdir()

    descriptor = _descriptor()
    descriptor["target_layers"] = ("layer.primary", "layer.alias0", "layer.alias1")
    _save_record(
        decode_dir / "d.pt",
        "consumer",
        descriptor,
        b"\x01\x02\x03\x04",
        logical_tensor_names=("layer.primary", "layer.alias0", "layer.alias1"),
    )
    _save_record(
        golden_dir / "g0.pt",
        "golden",
        {**descriptor, "target_layers": ("layer.alias0",)},
        b"\x01\x02\x03\x04",
        logical_tensor_names=("layer.alias0",),
    )
    _save_record(
        golden_dir / "g1.pt",
        "golden",
        {**descriptor, "target_layers": ("layer.alias1",)},
        b"\x01\x02\x00\x04",
        logical_tensor_names=("layer.alias1",),
    )

    golden_results = compare_decode_golden(
        load_records(decode_dir), load_records(golden_dir)
    )

    assert [result.key for result in golden_results] == [
        (0, "layer.alias0", 0),
        (0, "layer.alias1", 0),
    ]
    assert golden_results[0].allclose is True
    assert golden_results[1].max_abs is not None
    assert golden_results[1].max_abs > 0


def test_compare_xfer_debug_expands_descriptor_target_layers_without_record_aliases(
    tmp_path: Path,
):
    decode_dir = tmp_path / "decode"
    golden_dir = tmp_path / "golden"
    decode_dir.mkdir()
    golden_dir.mkdir()

    descriptor = _descriptor()
    descriptor["target_layers"] = ("layer.primary", "layer.alias0", "layer.alias1")
    _save_record(
        decode_dir / "d.pt",
        "consumer",
        descriptor,
        b"\x01\x02\x03\x04",
    )
    _save_record(
        golden_dir / "g0.pt",
        "golden",
        {**descriptor, "target_layers": ("layer.alias0",)},
        b"\x01\x02\x03\x04",
    )
    _save_record(
        golden_dir / "g1.pt",
        "golden",
        {**descriptor, "target_layers": ("layer.alias1",)},
        b"\x01\x02\x00\x04",
    )

    golden_results = compare_decode_golden(
        load_records(decode_dir), load_records(golden_dir)
    )

    assert [result.key for result in golden_results] == [
        (0, "layer.alias0", 0),
        (0, "layer.alias1", 0),
    ]
    assert golden_results[0].allclose is True
    assert golden_results[1].max_abs is not None
    assert golden_results[1].max_abs > 0


def _descriptor() -> dict[str, object]:
    return descriptor_to_dict(
        XferDebugDescriptor(
            transfer_id="transfer-1",
            d_req_id="d-req-1",
            tp_rank=0,
            descriptor_idx=0,
            region_idx=1,
            block_id=3,
            local_block_id=3,
            remote_block_id=3,
            local_block_ids=(3,),
            remote_block_ids=(3,),
            block_ordinals=(0,),
            src_ptr=0,
            dst_ptr=0,
            length=4,
            src_offset=0,
            dst_offset=0,
            source_layer="model.layers.2.attn",
            target_layers=("model.layers.2.attn",),
            source_block_len=4,
            target_block_len=4,
            bucket_type="37440",
        )
    )


class _FakeKVCacheTensor(NamedTuple):
    shared_by: tuple[str, ...]


class _FakeKVCacheGroup(NamedTuple):
    layer_names: tuple[str, ...]


class _FakeKVCacheConfig(NamedTuple):
    kv_cache_tensors: tuple[_FakeKVCacheTensor, ...]
    kv_cache_groups: tuple[_FakeKVCacheGroup, ...]


def _save_record(
    path: Path,
    side: str,
    descriptor: dict[str, object],
    payload: bytes,
    materialized_block_bytes: int | None = None,
    logical_tensor_names: tuple[str, ...] | None = None,
) -> None:
    payload_tensor = torch.tensor(list(payload), dtype=torch.uint8)
    record = {
        "kind": "mooncake_xfer_debug",
        "side": side,
        "pointer_kind": "source" if side != "consumer" else "destination",
        "descriptor": descriptor,
        "tensor_dtype": "torch.uint8",
        "payload": payload_tensor,
        "payload_num_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if materialized_block_bytes is not None:
        record["materialized_block_bytes"] = materialized_block_bytes
    if logical_tensor_names is not None:
        record["logical_tensor_names"] = logical_tensor_names
    torch.save(
        record,
        path,
    )
