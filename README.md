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

### 4. Opt-in routed-only expert streaming — `STRATA_Q36_ROUTE_STREAM=1`
The streamed prompt path used to copy **every** expert of a layer into VRAM
staging per chunk (256 experts × 40 layers ≈ 10.8 GB over PCIe for a 512-token
chunk) although a chunk routes to only ~150 distinct experts. This flag counts
the chunk's routes on the device (`k_route_count` / `k_routed_list`), copies just
those experts into a compact staging pair, and remaps the grouped kernels' blob
pointers (`k_remap_groups`) — ~4× less PCIe traffic and ~4× smaller staging.
`STRATA_Q36_ROUTE_CAP` (default 192) sets the compact capacity in experts; a
chunk routing more than the cap falls back to the CPU pool for that layer
(correct, slower). The one-session context fit check accounts for the compact
staging, so small cards get the right "tokens fit here" figure.

## Measured on GTX 1650 4 GB

Qwen3.6-35B-A3B UD-IQ3_XXS (13.2 GB pack), experts on CPU, dense weights on GPU:

| | upstream path | this fork |
|---|---|---|
| pack loads | refused (format gates) | yes, plain + MTP |
| decode | — | ~30.7 tok/s |
| max context | 7,168 (fp32 KV) | **32,768** (int8 KV + small scores buffer); **15,360 with fast prefill** (routed streaming staging must fit) |
| prefill (1500-token prompt) | ~45 tok/s | **~78 tok/s** (routed streaming, 15,360 ctx) |
| long-context quality | — | needle correct at 10,941 tokens; routed-streaming needle correct at 1,130 tokens via the OpenAI API |

Honest limits on this hardware:

- **Prefill without routed streaming stays slow (~45 tok/s):** the whole-layer
  weight stream dominates. With `STRATA_Q36_ROUTE_STREAM=1` it reaches ~78 tok/s
  while the compact staging fits (≤15,360 context on this card); beyond that the
  engine falls back to the CPU pool path. The remaining cost is the per-expert
  copy overhead and the CPU pool's decode-side work, not the expert math.
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
# recommended on this card: int8 KV + small scores buffer + routed-only streaming
# (15,360 context with ~78 tok/s prefill; cap 160 leaves staging headroom)
STRATA_Q36_KV_Q8=1 STRATA_Q36_SMALL_ATT=1 STRATA_Q36_ROUTE_STREAM=1 STRATA_Q36_ROUTE_CAP=160 \
  ./build-sm75/strata --serve \
  --model Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf --max-context 15360 --batch 1 --chunk 512

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