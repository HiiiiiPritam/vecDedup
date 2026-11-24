# VecDedup

**Semantic-aware snapshotting & deduplication for vector databases.** Git-style
versioning (commits, branches, diffs) for embedding sets *and* their HNSW indexes. It
stores each new snapshot as per-vector XOR deltas and per-node edge deltas instead
of byte chunks.

* `docs/RESEARCH.md`: literature review and what it changed in the spec (read first)
* `docs/DESIGN.md`: architecture, data model, algorithms, limitations
* `results/RESULTS.md`: measured results; `results/*.png` charts for slides
* `docs/LEARNING_NOTES.md`: beginner-friendly explanations of every concept and feature, with worked examples
* `docs/PPT_CONTENT.md`: slide-by-slide content, formulas, numbers, demo script and Q&A for the presentation
* `docs/STUDY_GUIDE.md`: compact single-file reference (good to give to a study chatbot)

## Quick start

Requirements: Go ≥ 1.22. Python 3 + matplotlib is needed only for the plots.

```powershell
go test ./...                                   # full test suite (~30 s)
go build -o bin/vecdedup.exe ./cmd/vecdedup
go build -o bin/vdbench.exe  ./cmd/vdbench

# demo
bin/vecdedup gen   --n 20000 --dim 128 --out v1.vdv
bin/vecdedup churn --in v1.vdv --op mixed --frac 0.05 --out v2.vdv
bin/vecdedup gen   --n 100 --dim 128 --seed 9 --out q.vdv
bin/vecdedup -C demo commit v1.vdv -m "base" --hnsw
bin/vecdedup -C demo commit v2.vdv -m "5% churn" --hnsw   # HNSW updated incrementally
bin/vecdedup -C demo branch experiment main~1             # zero-copy branch
bin/vecdedup -C demo log
bin/vecdedup -C demo diff main~1 main                     # vectors + ΔE+/ΔE−
bin/vecdedup -C demo search main --query q.vdv            # restored index, no rebuild
bin/vecdedup -C demo verify
bin/vecdedup -C demo stats                                # dedup ratio
```

Lossy mode: `commit data.vdv --lossy --pq-m 48` (PQ sub-spaces; default dim/8).

### Web demo UI

```powershell
go run ./cmd/vddemo        # then open http://localhost:8080
```

Drives the real engine: generate, churn, commit, branch, diff, search, verify, plus a
"Run full scenario" button and a results tab. It keeps its repo in a temp dir.

### Benchmarks

```powershell
# optional real data: first 100k SIFT1M vectors (52 MB, HTTP range request)
mkdir data/sift
curl.exe -L -r 0-51599999 -o data/sift/sift_base_100k.fvecs https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs
curl.exe -L -o data/sift/sift_query.fvecs https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs

bin/vdbench --quick          # 1-minute smoke run
bin/vdbench                  # full run (~20-25 min): results/results.json + RESULTS.md
python scripts/plot.py       # charts
```

## Repository layout

```
cmd/vecdedup     CLI
cmd/vdbench      evaluation harness (all baselines + scenarios)
internal/vec     dataset type, .vdv / .fvecs IO, distance kernels
internal/synth   embedding-like data + churn simulator
internal/codec   byte shuffle, XOR delta, zstd, varints
internal/quant   k-means, residual product quantizer, drift detection
internal/hnsw    deterministic HNSW (insert / delete+repair / search / dirty tracking)
internal/segment canonical hashing + full/delta encodings (vector, code, graph)
internal/cas     content-addressed packs + Pebble metadata (atomic commits)
internal/repo    Merkle-DAG snapshots, branches, commit/checkout/diff/verify
internal/cdc     FastCDC + fixed-block baselines
internal/metrics ground truth + Recall@k
scripts/plot.py  charts
```

## Team split (3 members)

Each member owns one subsystem end to end (code, tests, benchmark section, slides).
The interfaces between subsystems already exist (`segment.Encoded`, `repo.CommitOptions`,
`hnsw.Node`), so the three of you can work in parallel without blocking each other.

| | Member A: Vector dedup | Member B: Graph engine | Member C: Storage & system |
|---|---|---|---|
| **Owns** | `codec`, `quant`, vector + code parts of `segment`, `synth` | `hnsw`, graph part of `segment` (TGIS), `metrics` | `cas`, `repo`, `cdc`, CLI, `vdbench`, plots |
| **Must be able to explain** | float32 layout, byte-plane shuffle, XOR delta, why lossless tops out near 1.2–1.5× for new data, k-means, PQ / IVFADC, drift detection | HNSW Algorithms 1/2/4, level distribution, neighbour heuristic, delete-and-repair, why one insert touches ~M lists, TGIS delta format | content addressing, Merkle trees, Git's object model, LSM trees / Pebble vs RocksDB, crash-safe commit protocol, FastCDC Gear hash and normalized chunking |
| **Owns these results** | vector scenario table, lossy recall curve | HNSW snapshot table, SIFT recall validation | latency table, baseline methodology |
| **Extension work (if time)** | spherical-coordinate transform (Jina 2026) for new vectors; per-dimension shuffle | parallel HNSW build; hnswlib file importer | garbage collection; gRPC server; Linux ZFS comparison in WSL |

Suggested schedule: weeks 1–3 everyone reads `RESEARCH.md` and their own
subsystem's code and papers; weeks 4–10 extensions plus more datasets (e.g. real
sentence-transformer embeddings of a Wikipedia dump re-embedded twice, which turns
the "re-embed noise" scenario into measured reality); weeks 11–14 final benchmarks
and report; weeks 15–16 rehearsal.

## Presentation split (~18 min + Q&A)

| # | Slide | Speaker |
|---|---|---|
| 1 | Problem: RAG pipelines create a new embedding set + index every day | C |
| 2 | Where byte-level dedup works and where it fails (vector scenario chart) | C |
| 3 | Architecture: segments, content addressing, Merkle-DAG, branches | C |
| 4 | Lossless: exact dedup + XOR deltas, and the 1.2–1.5× ceiling for new data | A |
| 5 | Lossy: residual PQ, shared codebook, fingerprints, drift detection | A |
| 6 | Results: dedup table + recall/compression curve | A |
| 7 | HNSW in 1 slide, and why one insert rewrites ~M neighbour lists | B |
| 8 | TGIS: edge deltas (ΔE⁺, ΔE⁻) + delete-and-repair that minimises churn | B |
| 9 | Results: index snapshot cost vs FastCDC; restored index = identical answers | B |
| 10 | Live demo: commit → churn → commit → branch → diff → search → stats | C |
| 11 | Latency + crash safety + verify | C |
| 12 | Honest limitations & future work (everyone answers for their part) | all |

Q&A questions to prepare for:
* *"Isn't lossy mode just Faiss IVF-PQ?"* The quantizer is IVFADC (2011). What's new
  is using it inside a versioned content-addressed store: a shared codebook object,
  per-vector fingerprints to skip unchanged rows, delta-encoded code segments, and
  drift-triggered retraining.
* *"Why not Git-LFS / DVC / lakeFS / ZFS?"* They dedupe *files or blocks*. Look at
  the compaction and HNSW rows of the results: one reordering or one index update
  and they store almost everything again.
* *"Your synthetic data is fake."* The HNSW and lossy numbers are also measured on SIFT
  (real data), and the generator's low-rank cluster structure is documented in `synth`.
