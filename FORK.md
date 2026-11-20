# strata-small-gpu

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT) tuned to run
**Qwen3.6-35B-A3B on small machines** — developed and measured on a laptop with a
**GTX 1650 (4 GB VRAM, sm_75)**, Ryzen 7 5800HS, 22 GiB RAM.

Upstream Strata targets 12 GB+ cards. This fork makes the whole Unsloth Qwen3.6
GGUF range load on a 4 GB card and triples the usable context.

## What this fork adds (3 commits on top of upstream)

### 1. Full Unsloth Qwen3.6 pack support (`8b51b59`, `3abdb72`)
The q36 engine's expert-format whitelist refused several quant formats used by
Unsloth packs. Added, transcribed from llama.cpp's `vecdotq.cuh` (VDR 1):

- **IQ3_S / IQ3_XXS as down-projection formats** (`STRATA_D_FMTS`) — main layers of UD-IQ3_XXS packs.
- **Q2_K gate/up + Q3_K down** (`vec_dot_q2_K_q8_1`, `vec_dot_q3_K_q8_1`, `Fmt<10>`/`Fmt<11>`, `dq_q2_k`) — the MTP drafting block of Unsloth MTP-repo packs.
- **Optional `nextn_predict_layers` metadata** — plain (non-MTP) packs omit it; previously a hard load error.

Result: every Unsloth Qwen3.6-35B-A3B pack (plain and MTP repos) passes the load gates.

### 2. Opt-in int8 KV cache (`3f91583`, `STRATA_Q36_KV_Q8=1`)
Storage-only quantization of the attention K/V cache, following the Flash-Next
`kv_q8` scheme: int8 codes + one FP16 scale per 64-value group. Quantize on write
(`k_attn_prep`), dequantize on read (`k_attn_part` decode, `k_kv_f16` prefill);
the attention math itself is unchanged. Single-session (`--batch 1`).

Per-token KV cost drops from 44 KiB (fp32) to ~11.3 KiB.

## Measured on GTX 1650 4 GB (Qwen3.6-35B-A3B UD-IQ3_XXS, 13.2 GB pack)

| | upstream path | this fork |
|---|---|---|
| pack loads | refused (format gates) | yes, plain + MTP |
| decode | — | ~30.7 tok/s (MTP drafting: no gain on 4 GB, CPU-bound) |
| max context | 7,168 (fp32 KV) | **23,552** (int8 KV); 16,384 used daily |
| long-context quality | — | needle retrieval correct at 10,941 tokens |

Known limit: prefill is ~32–43 tok/s (prompt experts run per-token on the CPU —
no VRAM for batched GPU kernels). A batched CPU prefill path is the next target.

## Build (CUDA, sm_75)

```sh
cmake -G Ninja -S engine -B build-sm75 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75 --target strata -j8
```

Tested with CUDA 13.4 / gcc 16 on Arch Linux. Other architectures: set
`CMAKE_CUDA_ARCHITECTURES` accordingly (upstream release builds 75;86;89;120).

## Run with int8 KV

```sh
STRATA_Q36_KV_Q8=1 ./build-sm75/strata --serve \
  --model Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --max-context 16384 --batch 1 --chunk 512
```

Without `STRATA_Q36_KV_Q8` the engine behaves exactly as upstream (byte-identical
default path); all fork features are opt-in.

## License

MIT, as upstream. Fork modifications copyright 2026 Andrew Anter.