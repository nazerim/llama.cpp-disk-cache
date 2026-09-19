# Fork note: disk-backed prompt cache

Upstream: https://github.com/ggml-org/llama.cpp (this repo is a fork, kept for the
comparison view against `master`).

## What is in here

One local branch, `disk-cache-patch`, carrying upstream PR **#28092**
("server : add --cache-disk for a persistent prompt cache", author `yitizi`) applied
verbatim on top of the commit it was written against.

| | |
|---|---|
| base commit | `911f6cdc8ab8a530b2bee09ee61471a6f3178eeb` (master, 2026-09-18) |
| patch source | https://github.com/ggml-org/llama.cpp/pull/28092.diff |
| patch status upstream | **open, unmerged, zero reviews**, `mergeable_state: blocked`, last updated 2026-09-14 |
| diff applied | 8 files, +1,706/-71 as published (+1,354/-71 net of context in `git apply --stat`) |
| new flags | `--cache-dir PATH`, `--cache-dir-max N` (MiB; -1 unlimited, 0 disable) |

Nothing else is modified. No local fixups, no reformatting, no extra features — so the
"compare" view against upstream stays meaningful and a future rebase stays mechanical.

## Why this fork exists

Stock llama.cpp has no disk tier for the prompt cache: `-cram` is RAM-only, and on overflow
entries are dropped, never spilled. Automatic disk persistence was requested upstream in
#17107 and **closed as not planned**; #20697 asks for it again and #28092 implements it. We
need it, so we carry it as a branch until it merges (or does not).

## Build recipe that is known to work (Windows, RTX 3080 + RTX 5070 Ti)

```
cmake -C cuda-opts.cmake -S llama -B build-cuda -G "Ninja Multi-Config" ^
  -DGGML_CUDA=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build build-cuda --config Release --target llama-server -j 10      # ~14 min
```

`cuda-opts.cmake` (kept in this branch's build notes, not in the tree):

```
set(CMAKE_CUDA_ARCHITECTURES "86;120a" CACHE STRING "cuda archs" FORCE)
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4/bin/nvcc.exe"
    CACHE FILEPATH "nvcc" FORCE)
```

Four things that cost time to discover, all verified on the box described above:

* **Ninja, not the Visual Studio generator.** The VS generator routes CUDA compiles through
  MSBuild's `CUDA 13.4.targets` and fails with `The CUDA Toolkit directory '' does not exist`.
* **Pin the compiler.** Boxes with old CUDA toolkits installed make `find_package(CUDAToolkit)`
  resolve a stale one (v9.2 here), whose `host_config.h` then rejects MSVC 14.44.
* **Pin both architectures.** `native` detects one device; this host needs `86` (Ampere) *and*
  `120a` (Blackwell consumer) or the other card has no kernels. `120a` rather than `120f`, which
  trips a CMake regex bug in some versions.
* **Stage the CUDA runtime DLLs.** CUDA 13.4 keeps the x64 copies in `bin\x64`, not `bin\`. A
  missing `cublas64_13.dll` produces an exe that exits 53 with no message at all.

Toolchain used: CUDA 13.4 (nvcc V13.4.92) + MSVC 19.44 / VS 2022 Community + CMake 4.4.3,
driver 616.92. The CUDA Toolkit stopped bundling the Windows display driver in 13.1, so
installing it cannot disturb driver settings.

## Measured behaviour (this is the reason to care)

Hybrid DeltaNet model, MTP speculative draft, KV `q8_0`, RAM prompt cache forced off with
`--cache-ram 0` so no RAM tier can explain a hit. Entries are written when a prompt is
displaced by a new task on the slot; nothing is flushed at shutdown.

| prompt | cold prefill | entry | save (synchronous) | same prompt after a **restart** |
|---|---|---|---|---|
| 704 tok | 998.84 ms | 480.3 MiB | 443 ms | 159.22 ms, 700 reused |
| 10,297 tok | 7,829.91 ms | ~1.5 GiB | 1,415 ms | 329.19 ms, 20,533 reused *(20.5k case)* |
| 30,400 tok | 24,457 ms | 1,816.4 MiB | 1,750 ms | 304.61 ms, 30,396 reused |
| 1,129 tok (image prompt) | 2,014 ms | 488.0 MiB | 456 ms | 306.07 ms, 1,125 reused |
| 77k / 102k tok (real sessions) | 79 s / 105 s | 6.2 / 7.9 GiB | 5.2-19.4 s | `f_keep = 1.000`, 77,009 and 102,311 reused |

Entry size ≈ 448 MiB fixed + ~46 KiB per token at short lengths, and at long context the
**checkpoints dominate** (240 MiB to ~1.1 GiB each), which is what makes `-ctxcp` / `-cms` the
knobs that control cost. The save is synchronous and charged to the *next* request: ~1 s per GiB
measured on a 1.5 GiB entry, up to 49.9 s with upstream's default checkpoint budget.

Limits found by reading and by being bitten:

* `--cache-dir` **replaces** `--cache-ram`. There is no tiering; you get one or the other.
* Entries are keyed on model path/size/mtime, `n_ctx`, batch/ubatch, `n_parallel`, `-ctk`/`-ctv`,
  the spec-draft context, flash-attn, rope parameters, LoRAs, `has_mtmd`, **and the llama.cpp
  commit** — so a version bump or a profile change silently wipes the directory. Give each
  profile its own `--cache-dir`.
* Files are opened with `dwShareMode = 0` and mapped for the process's life; a second instance
  sharing a directory cannot read the first's metadata and **deletes those entries** in its
  startup sweep.
* Path handling uses the narrow-char `CreateFileA`/`CreateFileMappingA` on `std::string`, so
  non-ASCII cache paths are unreliable.
* If `--cache-dir-max` is below the size of one entry, the cache never fills and only warns.
* Prompt processing creates no checkpoints inside the prompt; the checkpoint positions a session
  accumulates come from where generation ran. `-cms` cannot place a checkpoint at an arbitrary
  boundary, so a brand-new session still pays full prefill once.
* Untested here: multi-slot (`-np > 1`) behaviour under contention, and disk-full/SIGBUS paths.

## Related upstream items

* #17107 — KV persistence to disk; **closed, not planned** (2025-11-08 → 2026-03-04).
* #20697 — open feature request for `--cache-disk`; #28092 closes it.
* #26004 — "preserve context checkpoints across slot save/restore"; the **same class of bug** in
  the older `--slot-save-path` path. It no longer applies to current master (stale, not a
  conflict) and the disk tier does not need it.
