# strata-small-gpu

**Run Qwen3.6-35B-A3B on a small laptop GPU.** A fork of
[Strata](https://github.com/Niko1221/Strata) (MIT) tuned for machines the upstream
engine doesn't target: developed and measured on a **GTX 1650 (4 GB VRAM, sm_75)**,
Ryzen 7 5800HS, 22 GiB RAM, fast NVMe.

Upstream Strata shines on 12 GB+ cards. Out of the box it **refused every Unsloth
Qwen3.6 GGUF pack** on this laptop (expert-format gates) and capped context at
7,168 tokens (fp32 KV cache). This fork fixes both, with all changes **opt-in** —
without the flags below the engine behaves exactly as upstream.

## What this fork changes

### 1. Full Unsloth Qwen3.6 pack support
The q36 engine's expert-format whitelist rejected quant formats used by Unsloth
packs. Added, transcribed from llama.cpp's `vecdotq.cuh` (VDR 1):

- **IQ3_S / IQ3_XXS as down-projection formats** (`STRATA_D_FMTS`) — the main
  layers of UD-IQ3_XXS packs.
- **Q2_K gate/up + Q3_K down** (`vec_dot_q2_K_q8_1`, `vec_dot_q3_K_q8_1`,
  `Fmt<10>`/`Fmt<11>`, `dq_q2_k`) — the MTP drafting block of Unsloth MTP-repo packs.
- **Optional `nextn_predict_layers` metadata** — plain (non-MTP) packs omit it;
  previously a hard load error.

Result: every Unsloth Qwen3.6-35B-A3B pack (plain and MTP repos) passes the load
gates on sm_75.

### 2. Opt-in int8 KV cache — `STRATA_Q36_KV_Q8=1`
Storage-only quantization of the attention K/V cache, following the Flash-Next
`kv_q8` scheme: int8 codes + one FP16 scale per 64-value group. Quantize on write
(`k_attn_prep`), dequantize on read (`k_attn_part` for decode, `k_kv_f16` for
prefill); the attention math itself stays full precision. Single-session
(`--batch 1`). Per-token KV cost drops from 44 KiB (fp32) to ~11.3 KiB.

### 3. Opt-in small prompt-scores buffer — `STRATA_Q36_SMALL_ATT=1`
The prompt attention scores buffer at 64 MB instead of 256 MB from init (the size
`shrink_att()` already uses when experts nearly fit). Long prompts take more,
smaller slices; frees ~192 MB of VRAM for the KV cache on small cards.

## Measured on GTX 1650 4 GB

Qwen3.6-35B-A3B UD-IQ3_XXS (13.2 GB pack), experts on CPU, dense weights on GPU:

| | upstream path | this fork |
|---|---|---|
| pack loads | refused (format gates) | yes, plain + MTP |
| decode | — | ~30.7 tok/s |
| max context | 7,168 (fp32 KV) | **32,768** (int8 KV + small scores buffer, desktop on the iGPU) |
| long-context quality | — | needle retrieval correct at 10,941 tokens |

Honest limits on this hardware:

- **Prefill is slow (~32–43 tok/s).** With ~no experts in VRAM, prompt experts run
  per-token on the CPU; there is no room for the batched GPU kernels. A batched
  CPU prefill path is the next target.
- **MTP drafting gives no speedup on 4 GB** (CPU-bound experts; draft overhead
  cancels the gain) and costs ~2K context. The plain pack is the better deal here;
  MTP support matters on bigger cards.

## Build (CUDA)

```sh
cmake -G Ninja -S engine -B build-sm75 -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_Q6K_EXPERTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75 --target strata -j8
```

Tested with CUDA 13.4 / gcc 16 on Arch Linux. Other cards: set
`CMAKE_CUDA_ARCHITECTURES` accordingly (upstream release builds 75;86;89;120).

## Run

```sh
# int8 KV + small scores buffer: 32K context on a 4 GB card (desktop on the iGPU)
STRATA_Q36_KV_Q8=1 STRATA_Q36_SMALL_ATT=1 ./build-sm75/strata --serve \
  --model Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --max-context 32768 --batch 1 --chunk 512

# without the flag: upstream behavior, fp32 KV (7168 tokens fit on this card)
./build-sm75/strata --serve \
  --model Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --max-context 7168 --batch 1 --chunk 512
```

The engine speaks Strata's line protocol; the upstream `serve/` control bundle
(OpenAI + Anthropic APIs on `127.0.0.1:8080`) works with this binary unchanged.
Everything else — models, installers, docs — is upstream's; see
[the Strata README](https://github.com/Niko1221/Strata#readme) and `docs/`.

## License

MIT, as upstream. Fork modifications copyright 2026 Andrew Anter.