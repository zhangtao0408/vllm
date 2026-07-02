# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class ByteComparison:
    key: tuple[str, str, int, int]
    equal: bool
    p_sha256: str
    d_sha256: str
    p_len: int
    d_len: int
    first_diff: int | None


@dataclass(frozen=True)
class GoldenComparison:
    key: tuple[int, int, int]
    bucket_type: str
    length_match: bool
    compared_bytes: int
    max_abs: float | None
    mean_abs: float | None
    cosine: float | None
    allclose: bool | None


def load_records(dump_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(dump_dir.rglob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(record, dict) and record.get("kind") == "mooncake_xfer_debug":
            record["path"] = str(path)
            records.append(record)
    return records


def compare_prefill_decode(
    prefill_records: Iterable[dict[str, Any]],
    decode_records: Iterable[dict[str, Any]],
) -> list[ByteComparison]:
    producer = {
        _descriptor_key(record): record
        for record in prefill_records
        if record.get("side") == "producer"
    }
    consumer = {
        _descriptor_key(record): record
        for record in decode_records
        if record.get("side") == "consumer"
    }
    comparisons: list[ByteComparison] = []
    for key in sorted(producer.keys() & consumer.keys()):
        p_bytes = _payload_bytes(producer[key])
        d_bytes = _payload_bytes(consumer[key])
        comparisons.append(
            ByteComparison(
                key=key,
                equal=p_bytes == d_bytes,
                p_sha256=str(producer[key]["payload_sha256"]),
                d_sha256=str(consumer[key]["payload_sha256"]),
                p_len=len(p_bytes),
                d_len=len(d_bytes),
                first_diff=_first_diff(p_bytes, d_bytes),
            )
        )
    missing_decode = sorted(producer.keys() - consumer.keys())
    missing_prefill = sorted(consumer.keys() - producer.keys())
    _print_missing("decode", missing_decode)
    _print_missing("prefill", missing_prefill)
    return comparisons


def compare_decode_golden(
    decode_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
) -> list[GoldenComparison]:
    golden_chunks = {
        key: chunk
        for record in golden_records
        if record.get("side") == "golden"
        for key, chunk in _iter_block_records(record)
    }
    comparisons: list[GoldenComparison] = []
    for decode_record in decode_records:
        if decode_record.get("side") != "consumer":
            continue
        for key, chunk in _iter_block_records(decode_record):
            golden_bytes = golden_chunks.get(key)
            if golden_bytes is None:
                continue
            compare_len = min(len(chunk), len(golden_bytes))
            stats = _numeric_stats(
                chunk[:compare_len],
                golden_bytes[:compare_len],
                str(decode_record.get("tensor_dtype", "")),
            )
            descriptor = decode_record["descriptor"]
            comparisons.append(
                GoldenComparison(
                    key=key,
                    bucket_type=str(descriptor["bucket_type"]),
                    length_match=len(chunk) == len(golden_bytes),
                    compared_bytes=compare_len,
                    max_abs=stats[0],
                    mean_abs=stats[1],
                    cosine=stats[2],
                    allclose=stats[3],
                )
            )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefill-dump", type=Path, required=True)
    parser.add_argument("--decode-dump", type=Path, required=True)
    parser.add_argument("--golden-dump", type=Path)
    args = parser.parse_args()

    prefill_records = load_records(args.prefill_dump)
    decode_records = load_records(args.decode_dump)
    byte_results = compare_prefill_decode(prefill_records, decode_records)
    _print_byte_results(byte_results)

    if args.golden_dump is not None:
        golden_records = load_records(args.golden_dump)
        golden_results = compare_decode_golden(decode_records, golden_records)
        _print_golden_results(golden_results)

    failed_bytes = any(not result.equal for result in byte_results)
    return 1 if failed_bytes else 0


def _descriptor_key(record: dict[str, Any]) -> tuple[str, str, int, int]:
    descriptor = record["descriptor"]
    return (
        str(descriptor["transfer_id"]),
        str(descriptor["d_req_id"]),
        int(descriptor["tp_rank"]),
        int(descriptor["descriptor_idx"]),
    )


def _iter_block_records(
    record: dict[str, Any],
) -> Iterable[tuple[tuple[int, int, int], bytes]]:
    descriptor = record["descriptor"]
    payload = _payload_bytes(record)
    block_ordinals = [int(item) for item in descriptor["block_ordinals"]]
    if not block_ordinals:
        return
    if len(block_ordinals) <= 1:
        yield (
            (
                int(descriptor["tp_rank"]),
                int(descriptor["region_idx"]),
                block_ordinals[0],
            ),
            payload,
        )
        return
    chunk_len = len(payload) // len(block_ordinals)
    for offset, block_ordinal in enumerate(block_ordinals):
        start = offset * chunk_len
        yield (
            (int(descriptor["tp_rank"]), int(descriptor["region_idx"]), block_ordinal),
            payload[start : start + chunk_len],
        )


def _payload_bytes(record: dict[str, Any]) -> bytes:
    payload = record["payload"]
    if isinstance(payload, torch.Tensor):
        payload = payload.detach().cpu().contiguous()
        try:
            return payload.numpy().tobytes()
        except RuntimeError:
            return bytes(payload.view(torch.uint8).tolist())
    if isinstance(payload, bytes):
        return payload
    return bytes(payload)


def _first_diff(left: bytes, right: bytes) -> int | None:
    for idx, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _numeric_stats(
    left: bytes,
    right: bytes,
    dtype_name: str,
) -> tuple[float | None, float | None, float | None, bool | None]:
    dtype = _torch_dtype(dtype_name)
    if dtype is None:
        return None, None, None, None
    element_size = torch.tensor([], dtype=dtype).element_size()
    usable_len = min(len(left), len(right))
    usable_len -= usable_len % element_size
    if usable_len == 0:
        return None, None, None, None
    try:
        left_tensor = torch.frombuffer(left[:usable_len], dtype=dtype).float()
        right_tensor = torch.frombuffer(right[:usable_len], dtype=dtype).float()
    except (TypeError, RuntimeError):
        return None, None, None, None
    diff = (left_tensor - right_tensor).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = _cosine(left_tensor, right_tensor)
    allclose = bool(torch.allclose(left_tensor, right_tensor, rtol=1e-2, atol=1e-2))
    return max_abs, mean_abs, cosine, allclose


def _torch_dtype(dtype_name: str) -> torch.dtype | None:
    if not dtype_name.startswith("torch."):
        return None
    dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denom = float(torch.linalg.norm(left).item() * torch.linalg.norm(right).item())
    if math.isclose(denom, 0.0):
        return None
    return float(torch.dot(left, right).item() / denom)


def _print_byte_results(results: list[ByteComparison]) -> None:
    print(f"P/D byte comparisons: {len(results)}")
    for result in results:
        status = "OK" if result.equal else "DIFF"
        print(
            f"{status} key={result.key} p_len={result.p_len} d_len={result.d_len} "
            f"p_sha256={result.p_sha256[:12]} d_sha256={result.d_sha256[:12]} "
            f"first_diff={result.first_diff}"
        )


def _print_golden_results(results: list[GoldenComparison]) -> None:
    print(f"D/golden numeric comparisons: {len(results)}")
    for result in results:
        print(
            f"key={result.key} bucket={result.bucket_type} "
            f"length_match={result.length_match} bytes={result.compared_bytes} "
            f"max_abs={result.max_abs} mean_abs={result.mean_abs} "
            f"cosine={result.cosine} allclose={result.allclose}"
        )


def _print_missing(side: str, keys: list[tuple[str, str, int, int]]) -> None:
    if keys:
        print(f"Missing {side} records for {len(keys)} descriptor(s): {keys[:8]}")


if __name__ == "__main__":
    raise SystemExit(main())
