# Mooncake KV Xfer Debug Dump

This debug path dumps KV-cache transfer slices as `torch.save` records. Use it
to check two things:

1. 950PR Prefill source bytes and H20 Decode received bytes are identical.
2. H20 Decode received bytes are numerically close to an H20 native prefill
   golden dump.

The dump is disabled by default. Set `max_requests` to `1` for normal debugging
so only the first request is captured.

## 1. Clean Dump Directories

Run this before each test to avoid comparing stale records:

```bash
rm -rf /tmp/pd_xfer_4l
mkdir -p /tmp/pd_xfer_4l/{prefill,decode,h20_golden,ascend_native}
```

## 2. Dump PD Transfer Slices

Use this when running 950PR as Prefill and H20 as Decode through
`MooncakeConnector`.

Prefill config:

```json
{
  "kv_connector": "MooncakeConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "shadow_buckets": "/path/to/h20_4l_shadow_buckets.json",
    "xfer_debug": {
      "dump_dir": "/tmp/pd_xfer_4l/prefill",
      "max_requests": 1,
      "dump_full_source_block": true
    }
  }
}
```

Decode config:

```json
{
  "kv_connector": "MooncakeConnector",
  "kv_role": "kv_consumer",
  "kv_connector_extra_config": {
    "xfer_debug": {
      "dump_dir": "/tmp/pd_xfer_4l/decode",
      "max_requests": 1
    }
  }
}
```

Then send one request with the same prompt that will be used for the golden run.

What this captures:

- Prefill dumps the source slice after transfer descriptors are built and before
  Mooncake sends the data.
- Decode dumps the destination slice after the OK response is received.
- `dump_full_source_block: true` also saves the full source block when the
  copied window is smaller than the 950PR block, for example `40960 -> 37440`
  or `40960 -> 1728`.

## 3. Dump H20 Native Prefill Golden

This is for the H20 mixed/native run. Do not configure `kv_connector` here.
Mixed mode does not enter `mooncake_connector.py`; the dump is enabled through
`GPUModelRunner` with an environment variable.

Create a config file:

```bash
cat >/tmp/h20_golden_xfer_debug.json <<'JSON'
{
  "dump_dir": "/tmp/pd_xfer_4l/h20_golden",
  "max_requests": 1,
  "dump_on_prefill": true
}
JSON
```

Start the normal H20 service with:

```bash
export VLLM_KV_XFER_DEBUG_CONFIG=/tmp/h20_golden_xfer_debug.json
```

Then send the same prompt as the PD run, using the same 4-layer model slice and
sampling settings. The dumped records use `side="golden"` and do not need to
have the same request id as the PD run.

With data parallelism, each DP worker applies `max_requests` independently.
Dump files include a `dpX__tpY` prefix and a `rank_tag` field so records from
DP4 runs can be separated after collection. File names also include
`tensor_<name>` with the logical KV cache name selected from the descriptor.
For shared physical tensors, source dumps prefer `source_layer`; destination
dumps prefer a matching name from `target_layers`.

## 4. Dump 950PR Native Prefill

This is for the 950PR mixed/native run through vLLM-Ascend. Do not configure
`kv_connector` here. vLLM-Ascend's native runner also reads
`VLLM_KV_XFER_DEBUG_CONFIG`.

Create a config file:

```bash
cat >/tmp/ascend_native_xfer_debug.json <<'JSON'
{
  "dump_dir": "/tmp/pd_xfer_4l/ascend_native",
  "max_requests": 1,
  "dump_on_prefill": true
}
JSON
```

Start the normal 950PR native service with:

```bash
export VLLM_KV_XFER_DEBUG_CONFIG=/tmp/ascend_native_xfer_debug.json
```

Then send the same prompt as the PD and H20 native runs, using the same 4-layer
model slice and sampling settings. This dump captures Ascend prefill KV after
the forward pass and before sampling. For tuple KV caches, the first tensor keeps
the original physical layer name and later tensors use `#tensorN` suffixes in
the debug view names.

## 5. Compare Dumps

Run from the vLLM repository root:

```bash
python examples/disaggregated/mooncake_connector/compare_xfer_debug.py \
  --prefill-dump /tmp/pd_xfer_4l/prefill \
  --decode-dump /tmp/pd_xfer_4l/decode \
  --golden-dump /tmp/pd_xfer_4l/h20_golden
```

The script prints:

- `P/D byte comparisons`: byte equality between 950PR source and H20 received
  descriptor payloads, including padded bytes when a KV block stride is larger
  than the materialized tensor bytes.
- `D/golden numeric comparisons`: numeric stats between H20 received slices and
  H20 native prefill golden slices. Padding bytes are trimmed per block using
  `materialized_block_bytes` before numeric comparison.

## 6. Result Reading

- `P/D` shows `DIFF`: first check descriptor address, offset, length, and
  Mooncake write path.
- `P/D` is all `OK`, but `D/golden` has large error: check shadow-bucket window
  layout or 950PR/H20 KV generation differences.
- Only `bucket=c128_1728` has large error: focus on which window is copied from
  the 950PR `40960` source block into the H20 compact C128 bucket.

Each `.pt` record contains the descriptor metadata, payload bytes,
`payload_sha256`, tensor name, tensor dtype/shape, rank tag, block stride bytes,
materialized block bytes, `payload_materialized_num_bytes`,
`payload_padding_num_bytes`, and optional `full_source_block`.

## 7. Build Synthetic H20 Shadow Slices From 950PR Native

Use this after collecting both `ascend_native` and `h20_golden`. It does not
need H20 or 950PR hardware; it only reads dumped tensors with `torch.load`.

```bash
python examples/disaggregated/mooncake_connector/synthesize_h20_shadow_from_ascend.py \
  --ascend-native-dump /tmp/pd_xfer_4l/ascend_native \
  --h20-golden-dump /tmp/pd_xfer_4l/h20_golden \
  --output-dump /tmp/pd_xfer_4l/synthetic_h20
```

The script emits one synthetic record per H20 native golden record and prints a
comparison table keyed by `(tp_rank, logical_tensor_name, block_ordinal)`.
Current built-in rules cover the 4-layer DSv4 debug slice:

- `8192 -> 8192` and `32768 -> 32768`: direct copy for compressor state caches.
- `8192 -> 8448`: diagnostic `indexer.k_cache` candidates for `128 -> 132`.
- `40960 -> 37376`: row-wise `64 x 640 -> 64 x 584` candidates plus a scanned
  best column offset for diagnosis.
- `40960 -> 1168`: C128 compact candidate `last2_first584`, which maps the last
  two `640`-byte rows to H20's `2 x 584` materialized payload.

Read the output as a conversion diagnosis:

- `allclose=True` means the candidate is a viable byte/numeric layout rule for
  that dumped prompt.
- high cosine with small `mean_abs` on float32 state cache usually means the
  source/target logical cache is correct but backend numeric differences remain.
- `nan=(left,right,mismatch)` flags NaNs in synthetic/source versus H20 golden;
  this usually means the source logical view or block group still needs checking.

For `40960 -> 37440` layout work, add `--field-heatmap`:

```bash
python examples/disaggregated/mooncake_connector/synthesize_h20_shadow_from_ascend.py \
  --ascend-native-dump /tmp/pd_xfer_4l/ascend_native \
  --h20-golden-dump /tmp/pd_xfer_4l/h20_golden \
  --field-heatmap
```

The heatmap splits H20's `584` materialized bytes into `nope=448`,
`rope=128`, and `scale=8`, then prints per-field error and the worst rows for
each `40960 -> 37440` candidate. Use this before collecting new server dumps.

To check SWA source mapping without assuming same-name mapping, add
`--swa-source-scan`:

```bash
python examples/disaggregated/mooncake_connector/synthesize_h20_shadow_from_ascend.py \
  --ascend-native-dump /tmp/pd_xfer_4l/ascend_native \
  --h20-golden-dump /tmp/pd_xfer_4l/h20_golden \
  --swa-source-scan \
  --top-k 8
```

This scans every 950PR `40960` fp8 source block against each H20
`*.swa_cache` target and prints the closest source physical name, source tensor
name, source block ordinal, source block id, and transform candidate.

The scan prints two SWA sections:

- `SWA_TARGET`: global numeric nearest sources across all 950PR `40960` fp8
  blocks. This is useful for discovering surprising candidates, but sparse
  SWA blocks can make an unrelated mostly-zero block look close.
- `SWA_STRUCTURAL_TARGET`: same-name structural candidates, for example H20
  `model.layers.2.attn.swa_cache` against 950PR
  `model.layers.2.self_attn.swa_cache`. Treat this as the first semantic
  mapping check before trusting the global nearest result.

## 8. Next Alignment Order

Use the current dump before collecting new server data. The preferred order is:

1. Continue offline analysis for `40960 -> 37440`.
   H20's `584` bytes per token means `448 NoPE + 128 RoPE + 8 fp8 scale`, so
   this should be treated as field-level repacking from 950PR's `64 x 640`
   block rather than as a contiguous prefix. First add row/field heatmaps to
   locate whether the remaining error is in NoPE, RoPE, scale, or every field.

2. Then locate `indexer.k_cache 8192 -> 8448`.
   H20's `132` bytes per token means `128 fp8 + 4 scale`. The extra
   `256` bytes per block are real scale bytes, not padding. If the current
   native dump does not contain the scale source, collect a minimal 950PR
   native dump that includes `indexer_k_cache`, `indexer_scale_cache`, and
   `indexer_full_cache` when present in the Ascend tuple cache.

3. Finally isolate NaNs in `layer3.compressor.state_cache`.
   Do not treat this as a numeric tolerance problem. First verify the 950PR
   source logical view, tuple component, block id, and block group. A minimal
   950PR native dump with per-component finite/NaN stats is enough; do not rerun
   the full H20+950PR+PD stack unless the source-side view is confirmed correct.

Only collect new server data after the offline `40960 -> 37440` analysis stops
making progress or after the missing `indexer` scale/full-cache source needs to
be proven from runtime tensors.
