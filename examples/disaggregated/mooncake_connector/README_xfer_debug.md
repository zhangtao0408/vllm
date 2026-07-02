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
mkdir -p /tmp/pd_xfer_4l/{prefill,decode,h20_golden}
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

## 4. Compare Dumps

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

## 5. Result Reading

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
