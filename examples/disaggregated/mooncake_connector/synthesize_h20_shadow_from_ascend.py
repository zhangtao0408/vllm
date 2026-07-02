# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import torch

H20_MLA_FIELDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("nope", 0, 448),
    ("rope", 448, 128),
    ("scale", 576, 8),
)
H20_MLA_ROWS: Final[int] = 64
H20_MLA_ROW_BYTES: Final[int] = 584


@dataclass(frozen=True)
class CandidateResult:
    name: str
    length_match: bool
    compared_bytes: int
    max_abs: float | None
    mean_abs: float | None
    cosine: float | None
    allclose: bool | None
    first_diff: int | None
    left_nan_count: int
    right_nan_count: int
    nan_mismatch_count: int


@dataclass(frozen=True)
class SynthesisResult:
    key: tuple[int, str, int]
    source_name: str | None
    default_rule: str | None
    candidates: tuple[CandidateResult, ...]
    synthetic_record: dict[str, Any] | None


@dataclass(frozen=True)
class FieldStats:
    name: str
    compared_bytes: int
    max_abs: float | None
    mean_abs: float | None
    cosine: float | None
    allclose: bool | None
    first_diff: int | None


@dataclass(frozen=True)
class RowStats:
    row: int
    max_abs: float | None
    mean_abs: float | None
    cosine: float | None
    first_diff: int | None


@dataclass(frozen=True)
class FieldHeatmapResult:
    key: tuple[int, str, int]
    source_name: str
    candidate_name: str
    fields: tuple[FieldStats, ...]
    worst_rows: tuple[RowStats, ...]


@dataclass(frozen=True)
class SwaSourceScanResult:
    target_key: tuple[int, str, int]
    target_name: str
    source_name: str
    source_tensor_name: str
    source_block_ordinal: int
    source_block_id: int
    candidate_name: str
    result: CandidateResult


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build synthetic H20-layout KV slices from 950PR native xfer debug "
            "dumps and compare them with H20 native golden records."
        )
    )
    parser.add_argument("--ascend-native-dump", type=Path, required=True)
    parser.add_argument("--h20-golden-dump", type=Path, required=True)
    parser.add_argument("--output-dump", type=Path)
    parser.add_argument(
        "--show-candidates",
        action="store_true",
        help="Print every candidate transform, not only the default and best.",
    )
    parser.add_argument(
        "--field-heatmap",
        action="store_true",
        help="Print field and worst-row stats for 40960-to-37440 candidates.",
    )
    parser.add_argument(
        "--swa-source-scan",
        action="store_true",
        help="Scan every 950PR 40960 fp8 source block against each H20 SWA cache.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of matches to print per target for source scans.",
    )
    args = parser.parse_args()

    source_records = _load_records(args.ascend_native_dump)
    golden_records = _load_records(args.h20_golden_dump)
    results = synthesize_records(source_records, golden_records)

    if args.output_dump is not None:
        _write_synthetic_records(args.output_dump, results)

    _print_results(results, show_candidates=args.show_candidates)
    if args.field_heatmap:
        _print_field_heatmaps(source_records, golden_records)
    if args.swa_source_scan:
        _print_swa_source_scan(source_records, golden_records, max(args.top_k, 1))

    missing = sum(1 for result in results if result.synthetic_record is None)
    return 1 if missing else 0


def synthesize_records(
    source_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
) -> list[SynthesisResult]:
    source_groups = _source_records_by_canonical_name(source_records)
    golden_groups = _golden_records_by_canonical_name(golden_records)
    results: list[SynthesisResult] = []

    for target_name in sorted(golden_groups):
        target_records = golden_groups[target_name]
        source_name = _source_name_for_target(target_name)
        source_group = source_groups.get(source_name, ())
        for ordinal_idx, golden_record in enumerate(target_records):
            key = _record_key(golden_record)
            source_record = (
                source_group[ordinal_idx] if ordinal_idx < len(source_group) else None
            )
            if source_record is None:
                results.append(
                    SynthesisResult(
                        key=key,
                        source_name=source_name,
                        default_rule=None,
                        candidates=(),
                        synthetic_record=None,
                    )
                )
                continue

            target_len = _payload_len(golden_record)
            target_bytes = _payload_bytes(golden_record)
            target_dtype = str(golden_record.get("tensor_dtype", ""))
            candidates = _candidate_payloads(
                source_record,
                golden_record,
                target_len,
                target_bytes,
                target_dtype,
            )
            candidate_results = tuple(
                _compare_candidate(name, payload, target_bytes, target_dtype)
                for name, payload in candidates
            )
            default_rule, synthetic_payload = candidates[0]
            synthetic_record = _make_synthetic_record(
                golden_record=golden_record,
                source_record=source_record,
                payload=synthetic_payload,
                rule=default_rule,
            )
            results.append(
                SynthesisResult(
                    key=key,
                    source_name=source_name,
                    default_rule=default_rule,
                    candidates=candidate_results,
                    synthetic_record=synthetic_record,
                )
            )

    return results


def build_field_heatmaps(
    source_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
) -> list[FieldHeatmapResult]:
    source_groups = _source_records_by_canonical_name(source_records)
    golden_groups = _golden_records_by_canonical_name(golden_records)
    heatmaps: list[FieldHeatmapResult] = []

    for target_name in sorted(golden_groups):
        target_records = golden_groups[target_name]
        if not (target_name.endswith(".attn") or target_name.endswith(".swa_cache")):
            continue
        source_name = _source_name_for_target(target_name)
        source_group = source_groups.get(source_name, ())
        for ordinal_idx, golden_record in enumerate(target_records):
            if _payload_len(golden_record) != H20_MLA_ROWS * H20_MLA_ROW_BYTES:
                continue
            if ordinal_idx >= len(source_group):
                continue
            source_record = source_group[ordinal_idx]
            if int(source_record.get("block_stride_bytes", 0) or 0) != 40960:
                continue
            target_bytes = _payload_bytes(golden_record)
            target_dtype = str(golden_record.get("tensor_dtype", ""))
            candidates = _candidate_payloads(
                source_record,
                golden_record,
                len(target_bytes),
                target_bytes,
                target_dtype,
            )
            for candidate_name, candidate_payload in _heatmap_candidates(candidates):
                heatmaps.append(
                    FieldHeatmapResult(
                        key=_record_key(golden_record),
                        source_name=source_name,
                        candidate_name=candidate_name,
                        fields=_field_stats(
                            candidate_payload, target_bytes, target_dtype
                        ),
                        worst_rows=_worst_row_stats(
                            candidate_payload, target_bytes, target_dtype, limit=5
                        ),
                    )
                )

    return heatmaps


def scan_swa_sources(
    source_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, str, int], list[SwaSourceScanResult]]:
    source_candidates = tuple(_iter_swa_source_candidates(source_records))
    target_records = tuple(_iter_h20_swa_targets(golden_records))
    results: dict[tuple[int, str, int], list[SwaSourceScanResult]] = {}

    for target_record in target_records:
        target_key = _record_key(target_record)
        target_results: list[SwaSourceScanResult] = []
        for source_record in source_candidates:
            target_results.extend(
                _build_swa_source_results(source_record, target_record)
            )
        target_results.sort(key=_source_scan_score)
        results[target_key] = target_results

    return results


def scan_structural_swa_sources(
    source_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
) -> dict[tuple[int, str, int], list[SwaSourceScanResult]]:
    source_groups = _source_records_by_canonical_name(source_records)
    results: dict[tuple[int, str, int], list[SwaSourceScanResult]] = {}

    for target_record in _iter_h20_swa_targets(golden_records):
        target_name = _canonical_tensor_name(str(target_record.get("tensor_name", "")))
        source_group = source_groups.get(target_name, ())
        target_results: list[SwaSourceScanResult] = []
        for source_record in source_group:
            if int(source_record.get("block_stride_bytes", 0) or 0) != 40960:
                continue
            if _payload_len(source_record) != 40960:
                continue
            target_results.extend(
                _build_swa_source_results(source_record, target_record)
            )
        target_results.sort(key=_source_scan_score)
        results[_record_key(target_record)] = target_results

    return results


def _build_swa_source_results(
    source_record: dict[str, Any],
    target_record: dict[str, Any],
) -> list[SwaSourceScanResult]:
    target_key = _record_key(target_record)
    target_name = _canonical_tensor_name(str(target_record.get("tensor_name", "")))
    target_bytes = _payload_bytes(target_record)
    target_dtype = str(target_record.get("tensor_dtype", ""))
    results: list[SwaSourceScanResult] = []

    candidates = _candidate_payloads(
        source_record,
        target_record,
        len(target_bytes),
        target_bytes,
        target_dtype,
    )
    for candidate_name, candidate_payload in _heatmap_candidates(candidates):
        candidate_result = _compare_candidate(
            candidate_name, candidate_payload, target_bytes, target_dtype
        )
        results.append(
            SwaSourceScanResult(
                target_key=target_key,
                target_name=target_name,
                source_name=_source_display_name(source_record),
                source_tensor_name=str(source_record.get("tensor_name", "")),
                source_block_ordinal=_first_block_ordinal(source_record),
                source_block_id=_block_id(source_record),
                candidate_name=candidate_name,
                result=candidate_result,
            )
        )

    return results


def _iter_swa_source_candidates(
    records: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for record in records:
        if record.get("side") != "golden":
            continue
        if int(record.get("block_stride_bytes", 0) or 0) != 40960:
            continue
        if _payload_len(record) != 40960:
            continue
        yield record


def _iter_h20_swa_targets(
    records: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for record in records:
        if record.get("side") != "golden":
            continue
        tensor_name = record.get("tensor_name")
        if not isinstance(tensor_name, str):
            continue
        if not tensor_name.endswith(".swa_cache"):
            continue
        if _payload_len(record) != H20_MLA_ROWS * H20_MLA_ROW_BYTES:
            continue
        yield record


def _load_records(dump_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(dump_dir.rglob("*.pt")):
        record = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(record, dict) and record.get("kind") == "mooncake_xfer_debug":
            record["path"] = str(path)
            records.append(record)
    return records


def _source_records_by_canonical_name(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("side") != "golden":
            continue
        name = record.get("physical_tensor_name") or record.get("tensor_name")
        if not isinstance(name, str):
            continue
        groups[_canonical_tensor_name(name)].append(record)
    for grouped_records in groups.values():
        grouped_records.sort(key=_record_sort_key)
    return groups


def _golden_records_by_canonical_name(
    records: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("side") != "golden":
            continue
        name = record.get("tensor_name")
        if not isinstance(name, str):
            continue
        groups[_canonical_tensor_name(name)].append(record)
    for grouped_records in groups.values():
        grouped_records.sort(key=_record_sort_key)
    return groups


def _source_name_for_target(target_name: str) -> str:
    return target_name


def _candidate_payloads(
    source_record: dict[str, Any],
    golden_record: dict[str, Any],
    target_len: int,
    target_bytes: bytes,
    target_dtype: str,
) -> tuple[tuple[str, bytes], ...]:
    source = _payload_bytes(source_record)
    target_name = _canonical_tensor_name(str(golden_record.get("tensor_name", "")))
    source_stride = int(source_record.get("block_stride_bytes", 0) or 0)
    target_stride = int(golden_record.get("block_stride_bytes", 0) or 0)

    if target_name.endswith(".indexer.k_cache") and target_len == 8448:
        return (
            ("row_128_plus_zero4", _row_pack_with_zero_pad(source, 64, 128, 132)),
            ("tail_zero_8192_to_8448", _right_pad(source, target_len)),
            ("contiguous_prefix", _prefix(source, target_len)),
        )

    if (
        (target_name.endswith(".attn") or target_name.endswith(".swa_cache"))
        and source_stride == 40960
        and target_len == 37376
    ):
        return (
            ("row_first_584", _row_slice(source, 64, 640, 0, 584)),
            ("row_last_584", _row_slice(source, 64, 640, 56, 584)),
            ("contiguous_prefix", _prefix(source, target_len)),
            _best_column_slice_candidate(source, target_bytes, target_dtype),
        )

    if target_stride == 1728 and source_stride == 40960 and target_len == 1168:
        return (
            (
                "last2_first584",
                _row_slice(source, 64, 640, 0, 584, start_row=62, rows=2),
            ),
            ("first2_first584", _row_slice(source, 64, 640, 0, 584, rows=2)),
            ("contiguous_prefix", _prefix(source, target_len)),
        )

    if len(source) >= target_len:
        return (("direct_prefix", _prefix(source, target_len)),)

    return (("right_zero_pad", _right_pad(source, target_len)),)


def _make_synthetic_record(
    golden_record: dict[str, Any],
    source_record: dict[str, Any],
    payload: bytes,
    rule: str,
) -> dict[str, Any]:
    payload_tensor = _uint8_tensor_from_bytes(payload)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    descriptor = dict(golden_record["descriptor"])
    descriptor["source_layer"] = source_record.get("physical_tensor_name")
    descriptor["target_layers"] = (golden_record.get("tensor_name"),)
    return {
        "kind": "mooncake_xfer_debug",
        "side": "synthetic",
        "pointer_kind": "destination",
        "descriptor": descriptor,
        "tensor_dtype": golden_record.get("tensor_dtype"),
        "tensor_shape": golden_record.get("tensor_shape"),
        "tensor_name": golden_record.get("tensor_name"),
        "physical_tensor_name": golden_record.get("physical_tensor_name"),
        "logical_tensor_names": (golden_record.get("tensor_name"),),
        "block_stride_bytes": golden_record.get("block_stride_bytes"),
        "materialized_block_bytes": golden_record.get("materialized_block_bytes"),
        "rank_tag": golden_record.get("rank_tag"),
        "payload": payload_tensor,
        "payload_num_bytes": len(payload),
        "payload_materialized_num_bytes": len(payload),
        "payload_padding_num_bytes": 0,
        "payload_sha256": payload_sha256,
        "synthetic_source_path": source_record.get("path"),
        "synthetic_source_tensor_name": source_record.get("physical_tensor_name"),
        "synthetic_rule": rule,
    }


def _write_synthetic_records(
    output_dir: Path,
    results: Iterable[SynthesisResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, result in enumerate(results):
        record = result.synthetic_record
        if record is None:
            continue
        tensor_name = _safe_name(str(record.get("tensor_name", "unknown")))
        path = output_dir / f"synthetic__{idx:05d}__tensor_{tensor_name}.pt"
        torch.save(record, path)


def _compare_candidate(
    name: str,
    left: bytes,
    right: bytes,
    dtype_name: str,
) -> CandidateResult:
    compare_len = min(len(left), len(right))
    max_abs, mean_abs, cosine, allclose = _numeric_stats(
        left[:compare_len], right[:compare_len], dtype_name
    )
    left_nan_count, right_nan_count, nan_mismatch_count = _nan_stats(
        left[:compare_len], right[:compare_len], dtype_name
    )
    return CandidateResult(
        name=name,
        length_match=len(left) == len(right),
        compared_bytes=compare_len,
        max_abs=max_abs,
        mean_abs=mean_abs,
        cosine=cosine,
        allclose=allclose,
        first_diff=_first_diff(left, right),
        left_nan_count=left_nan_count,
        right_nan_count=right_nan_count,
        nan_mismatch_count=nan_mismatch_count,
    )


def _heatmap_candidates(
    candidates: tuple[tuple[str, bytes], ...],
) -> tuple[tuple[str, bytes], ...]:
    selected: list[tuple[str, bytes]] = []
    for name, payload in candidates:
        if name in {
            "row_first_584",
            "row_last_584",
            "contiguous_prefix",
        } or name.startswith("scan_best_col_offset_"):
            selected.append((name, payload))
    return tuple(selected)


def _field_stats(
    left: bytes,
    right: bytes,
    dtype_name: str,
) -> tuple[FieldStats, ...]:
    stats: list[FieldStats] = []
    for field_name, start_col, width in H20_MLA_FIELDS:
        left_field = _collect_field_bytes(left, start_col, width)
        right_field = _collect_field_bytes(right, start_col, width)
        compare_len = min(len(left_field), len(right_field))
        max_abs, mean_abs, cosine, allclose = _numeric_stats(
            left_field[:compare_len], right_field[:compare_len], dtype_name
        )
        stats.append(
            FieldStats(
                name=field_name,
                compared_bytes=compare_len,
                max_abs=max_abs,
                mean_abs=mean_abs,
                cosine=cosine,
                allclose=allclose,
                first_diff=_first_diff(left_field, right_field),
            )
        )
    return tuple(stats)


def _worst_row_stats(
    left: bytes,
    right: bytes,
    dtype_name: str,
    limit: int,
) -> tuple[RowStats, ...]:
    rows: list[RowStats] = []
    for row in range(H20_MLA_ROWS):
        start = row * H20_MLA_ROW_BYTES
        end = start + H20_MLA_ROW_BYTES
        left_row = left[start:end]
        right_row = right[start:end]
        compare_len = min(len(left_row), len(right_row))
        max_abs, mean_abs, cosine, _ = _numeric_stats(
            left_row[:compare_len], right_row[:compare_len], dtype_name
        )
        rows.append(
            RowStats(
                row=row,
                max_abs=max_abs,
                mean_abs=mean_abs,
                cosine=cosine,
                first_diff=_first_diff(left_row, right_row),
            )
        )
    rows.sort(key=_row_score, reverse=True)
    return tuple(rows[:limit])


def _collect_field_bytes(payload: bytes, start_col: int, width: int) -> bytes:
    chunks: list[bytes] = []
    for row in range(H20_MLA_ROWS):
        row_start = row * H20_MLA_ROW_BYTES
        start = row_start + start_col
        end = start + width
        if end <= len(payload):
            chunks.append(payload[start:end])
        else:
            chunks.append(bytes(width))
    return b"".join(chunks)


def _row_score(row: RowStats) -> tuple[float, float]:
    mean_score = _sortable_float(row.mean_abs)
    max_score = _sortable_float(row.max_abs)
    return (mean_score, max_score)


def _sortable_float(value: float | None) -> float:
    if value is None or math.isnan(value):
        return float("-inf")
    return value


def _source_scan_score(result: SwaSourceScanResult) -> tuple[float, float, int]:
    candidate = result.result
    first_diff = candidate.first_diff if candidate.first_diff is not None else -1
    return (
        float("inf") if candidate.mean_abs is None else candidate.mean_abs,
        float("inf") if candidate.max_abs is None else candidate.max_abs,
        first_diff,
    )


def _source_display_name(record: dict[str, Any]) -> str:
    name = record.get("physical_tensor_name") or record.get("tensor_name")
    return _canonical_tensor_name(str(name))


def _first_block_ordinal(record: dict[str, Any]) -> int:
    descriptor = record["descriptor"]
    ordinals = descriptor.get("block_ordinals") or (descriptor.get("block_id", 0),)
    return int(ordinals[0])


def _block_id(record: dict[str, Any]) -> int:
    descriptor = record["descriptor"]
    return int(descriptor.get("block_id", 0))


def _best_column_slice_candidate(
    source: bytes,
    target: bytes,
    dtype_name: str,
) -> tuple[str, bytes]:
    best_offset = 0
    best_payload = _row_slice(source, 64, 640, 0, 584)
    best_score = _score_payload(best_payload, target, dtype_name)
    for offset in range(1, 57):
        payload = _row_slice(source, 64, 640, offset, 584)
        score = _score_payload(payload, target, dtype_name)
        if score < best_score:
            best_offset = offset
            best_payload = payload
            best_score = score
    return (f"scan_best_col_offset_{best_offset}", best_payload)


def _score_payload(
    left: bytes,
    right: bytes,
    dtype_name: str,
) -> tuple[float, float]:
    max_abs, mean_abs, _, _ = _numeric_stats(left, right, dtype_name)
    mean_score = float("inf") if mean_abs is None or math.isnan(mean_abs) else mean_abs
    max_score = float("inf") if max_abs is None or math.isnan(max_abs) else max_abs
    return (mean_score, max_score)


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
        left_tensor = torch.frombuffer(
            bytearray(left[:usable_len]), dtype=dtype
        ).float()
        right_tensor = torch.frombuffer(
            bytearray(right[:usable_len]), dtype=dtype
        ).float()
    except (TypeError, RuntimeError):
        return None, None, None, None
    diff = (left_tensor - right_tensor).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    cosine = _cosine(left_tensor, right_tensor)
    allclose = bool(torch.allclose(left_tensor, right_tensor, rtol=1e-2, atol=1e-2))
    return max_abs, mean_abs, cosine, allclose


def _nan_stats(
    left: bytes,
    right: bytes,
    dtype_name: str,
) -> tuple[int, int, int]:
    dtype = _torch_dtype(dtype_name)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return (0, 0, 0)
    element_size = torch.tensor([], dtype=dtype).element_size()
    usable_len = min(len(left), len(right))
    usable_len -= usable_len % element_size
    if usable_len == 0:
        return (0, 0, 0)
    try:
        left_tensor = torch.frombuffer(bytearray(left[:usable_len]), dtype=dtype)
        right_tensor = torch.frombuffer(bytearray(right[:usable_len]), dtype=dtype)
    except (TypeError, RuntimeError):
        return (0, 0, 0)
    left_nan = torch.isnan(left_tensor)
    right_nan = torch.isnan(right_tensor)
    return (
        int(left_nan.sum().item()),
        int(right_nan.sum().item()),
        int((left_nan != right_nan).sum().item()),
    )


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


def _row_slice(
    source: bytes,
    source_rows: int,
    source_cols: int,
    col_offset: int,
    target_cols: int,
    *,
    start_row: int = 0,
    rows: int | None = None,
) -> bytes:
    rows = source_rows if rows is None else rows
    view = memoryview(source)
    chunks: list[bytes] = []
    for row in range(start_row, start_row + rows):
        row_start = row * source_cols
        start = row_start + col_offset
        end = start + target_cols
        if row < 0 or row >= source_rows or end > len(view):
            chunks.append(bytes(target_cols))
        else:
            chunks.append(bytes(view[start:end]))
    return b"".join(chunks)


def _row_pack_with_zero_pad(
    source: bytes,
    source_rows: int,
    source_cols: int,
    target_cols: int,
) -> bytes:
    view = memoryview(source)
    pad_cols = target_cols - source_cols
    if pad_cols < 0:
        raise ValueError("target_cols must be >= source_cols")
    chunks: list[bytes] = []
    for row in range(source_rows):
        start = row * source_cols
        end = start + source_cols
        if end > len(view):
            chunks.append(bytes(source_cols))
        else:
            chunks.append(bytes(view[start:end]))
        chunks.append(bytes(pad_cols))
    return b"".join(chunks)


def _prefix(source: bytes, length: int) -> bytes:
    if len(source) >= length:
        return source[:length]
    return _right_pad(source, length)


def _right_pad(source: bytes, length: int) -> bytes:
    if len(source) >= length:
        return source[:length]
    return source + bytes(length - len(source))


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


def _payload_len(record: dict[str, Any]) -> int:
    payload_num_bytes = record.get("payload_num_bytes")
    if isinstance(payload_num_bytes, int) and payload_num_bytes > 0:
        return payload_num_bytes
    return len(_payload_bytes(record))


def _record_key(record: dict[str, Any]) -> tuple[int, str, int]:
    descriptor = record["descriptor"]
    ordinals = descriptor.get("block_ordinals") or (descriptor.get("block_id"),)
    return (
        int(descriptor.get("tp_rank", 0)),
        _canonical_tensor_name(str(record.get("tensor_name", ""))),
        int(ordinals[0]),
    )


def _record_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    descriptor = record.get("descriptor", {})
    ordinals = descriptor.get("block_ordinals") or (descriptor.get("block_id", 0),)
    return (
        int(ordinals[0]),
        int(descriptor.get("block_id", 0)),
        str(record.get("path", "")),
    )


def _canonical_tensor_name(tensor_name: str) -> str:
    return tensor_name.replace(".self_attn.attn", ".attn").replace(
        ".self_attn.", ".attn."
    )


def _first_diff(left: bytes, right: bytes) -> int | None:
    for idx, (left_byte, right_byte) in enumerate(zip(left, right)):
        if left_byte != right_byte:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _uint8_tensor_from_bytes(raw: bytes) -> torch.Tensor:
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _print_results(
    results: Iterable[SynthesisResult],
    *,
    show_candidates: bool,
) -> None:
    results = list(results)
    print(f"Synthetic/H20 golden comparisons: {len(results)}")
    for result in results:
        if result.synthetic_record is None:
            print(f"MISSING key={result.key} source={result.source_name}")
            continue
        default = result.candidates[0]
        best = min(
            result.candidates,
            key=lambda item: (
                float("inf") if item.mean_abs is None else item.mean_abs,
                float("inf") if item.max_abs is None else item.max_abs,
            ),
        )
        print(
            f"key={result.key} source={result.source_name} "
            f"default={_format_candidate(default)} "
            f"best={_format_candidate(best)}"
        )
        if show_candidates:
            for candidate in result.candidates:
                print(f"  candidate={_format_candidate(candidate)}")


def _print_field_heatmaps(
    source_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
) -> None:
    heatmaps = build_field_heatmaps(source_records, golden_records)
    print(f"Field heatmaps for 40960->37440 candidates: {len(heatmaps)}")
    for heatmap in heatmaps:
        print(
            f"HEATMAP key={heatmap.key} source={heatmap.source_name} "
            f"candidate={heatmap.candidate_name} "
            f"fields={_format_field_stats(heatmap.fields)} "
            f"worst_rows={_format_row_stats(heatmap.worst_rows)}"
        )


def _print_swa_source_scan(
    source_records: Iterable[dict[str, Any]],
    golden_records: Iterable[dict[str, Any]],
    top_k: int,
) -> None:
    source_records = tuple(source_records)
    golden_records = tuple(golden_records)

    scan_results = scan_swa_sources(source_records, golden_records)
    total = sum(len(results[:top_k]) for results in scan_results.values())
    print(f"SWA source scan global top_k={top_k} printed={total}")
    for target_key, results in sorted(scan_results.items()):
        print(f"SWA_TARGET key={target_key}")
        for rank, result in enumerate(results[:top_k], start=1):
            print(_format_swa_source_result(rank, result))

    structural_results = scan_structural_swa_sources(source_records, golden_records)
    structural_total = sum(
        len(results[:top_k]) for results in structural_results.values()
    )
    print(f"SWA structural same-name top_k={top_k} printed={structural_total}")
    for target_key, results in sorted(structural_results.items()):
        print(f"SWA_STRUCTURAL_TARGET key={target_key}")
        if not results:
            print("  missing same-name 950PR 40960 source")
            continue
        for rank, result in enumerate(results[:top_k], start=1):
            print(_format_swa_source_result(rank, result))


def _format_swa_source_result(rank: int, result: SwaSourceScanResult) -> str:
    candidate = result.result
    return (
        f"  rank={rank} target={result.target_name} "
        f"source={result.source_name} "
        f"source_tensor={result.source_tensor_name} "
        f"src_ord={result.source_block_ordinal} "
        f"src_block={result.source_block_id} "
        f"candidate={result.candidate_name} "
        f"mean={candidate.mean_abs} max={candidate.max_abs} "
        f"cosine={candidate.cosine} allclose={candidate.allclose} "
        f"first_diff={candidate.first_diff}"
    )


def _format_candidate(candidate: CandidateResult) -> str:
    text = (
        f"{candidate.name} length_match={candidate.length_match} "
        f"bytes={candidate.compared_bytes} max_abs={candidate.max_abs} "
        f"mean_abs={candidate.mean_abs} cosine={candidate.cosine} "
        f"allclose={candidate.allclose} first_diff={candidate.first_diff}"
    )
    if candidate.left_nan_count or candidate.right_nan_count:
        text += (
            f" nan=({candidate.left_nan_count},{candidate.right_nan_count},"
            f"{candidate.nan_mismatch_count})"
        )
    return text


def _format_field_stats(fields: tuple[FieldStats, ...]) -> str:
    return ";".join(
        (
            f"{field.name}:bytes={field.compared_bytes},"
            f"mean={field.mean_abs},max={field.max_abs},"
            f"cos={field.cosine},diff={field.first_diff}"
        )
        for field in fields
    )


def _format_row_stats(rows: tuple[RowStats, ...]) -> str:
    return ";".join(
        (
            f"{row.row}:mean={row.mean_abs},max={row.max_abs},"
            f"cos={row.cosine},diff={row.first_diff}"
        )
        for row in rows
    )


if __name__ == "__main__":
    raise SystemExit(main())
