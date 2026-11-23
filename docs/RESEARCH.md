# VecDedup — Research Notes

These notes record what the literature says about each claim in the original spec
(`VECDEDUP_SPECIFICATION.md`) and how each finding changed the design. Read this
before the design doc: it explains *why* the implementation differs from the spec.

---

## 1. Content-Defined Chunking (the baseline we compare against)

**FastCDC** (Xia et al., USENIX ATC 2016; extended in IEEE TPDS 2020).

* Rolling hash is **Gear**: `fp = (fp << 1) + G[b]`, where `G` is a table of 256
  random 64-bit values. One shift + one add + one table lookup per byte, which is
  why it is 3–12× faster than Rabin.
* **Cut-point test**: `fp & mask == 0`. Zero bits are padded into the mask so the
  effective window is wider (dedup ratio ≈ Rabin) while hashing stays cheap.
* **Normalized chunking**: a *stricter* mask `MaskS` (more 1-bits) is used before the
  target average size and a *looser* mask `MaskL` (fewer 1-bits) after it. This pulls
  the chunk-size distribution toward the average.
* **Sub-minimum skipping**: no cut point is checked before `MinSize`.

Typical parameters: min = avg/4, avg = 8 KiB, max = avg·8.
Implemented in `internal/cdc` (2016 variant, normalization level 1… with the
`MaskS` = avg bits + 1, `MaskL` = avg bits − 1 convention).

Sources: FastCDC USENIX ATC'16 paper; IEEE TPDS 2020; arXiv:2409.06066
("A Thorough Investigation of Content-Defined Chunking Algorithms").

## 2. When does byte-level dedup actually fail on vectors?

The spec claims FastCDC gets "0% dedup" on vector snapshots. **This is not true in
general**, and an evaluator who knows dedup will spot it:

| Churn scenario | What FastCDC sees | Predicted | **Measured** (incremental ratio, 50k×384, see `results/RESULTS.md`) |
|---|---|---|---|
| 1% of rows re-embedded in place | Unchanged rows are byte-identical | High | **9.7×** (VecDedup 94.8×) |
| 5 / 10 / 25% in place | One changed 1.5 KB row spoils its whole ~8 KB chunk | Medium | **2.4× / 1.5× / 1.1×** (VecDedup 19.8× / 10.0× / 4.0×) |
| Rows deleted + compacted / re-sorted | Row order changes | Medium → low | **1.0×** (VecDedup 1681×, since a reorder is free) |
| **Whole corpus re-embedded with last-bit noise** | Every float differs | ≈ 0 | **1.0×** (VecDedup 4.6× via XOR deltas) |
| **HNSW index after incremental update** | Links rewritten all over the file | Low | 1% churn: FastCDC adds **10.3 MiB**, VecDedup **0.3 MiB** |
| New embedding model | Every vector different | 0 for everyone | 1.0× (VecDedup lossless 1.2×, the literature ceiling) |

Lesson: our prediction that FastCDC is "fine on in-place edits" was only true at very
low churn. Since a chunk spans several rows, the fraction of spoiled chunks is
≈ 1 − (1−p)^(rows per chunk), which grows quickly with churn p.

Why re-embedding gives different bits: floating-point addition is not associative,
and GPU kernels pick different reduction orders/tilings depending on batch shape,
so the same sentence embedded in a different batch yields last-bit differences.
Bit-level comparisons of `all-MiniLM-L6-v2` across x86/ARM also differ. Even with
seeds and `cudnn.deterministic`, full determinism isn't guaranteed.

**Design consequence:** we benchmark *each scenario separately* and report honestly
where FastCDC is already fine (in-place edits) and where it collapses
(re-embedding noise, compaction/reorder, HNSW files). VecDedup's real wins are the
latter cases.

Sources: arXiv:2506.09501, arXiv:2512.22280 (Valori), arXiv:2601.06118, Milvus/Zilliz
FAQ on sentence-transformers non-determinism, OpenAI community thread on embedding
determinism.

## 3. Lossless compression of float32 embeddings — the hard ceiling

* Float32 = 1 sign + 8 exponent + 23 mantissa bits. For embeddings the mantissa is
  close to **maximum entropy**, so only the sign/exponent bytes compress.
* Prior best (ZipNN-style transpose + byte-shuffle + entropy coder): **≈1.2×**.
* Jina AI's spherical-coordinate method (arXiv:2602.00079, 2026): **≈1.5×**, and the
  later versions describe it as error < 1e-7 rather than strictly bit-exact.
* Gorilla/Chimp XOR coding on embeddings: ~15% gain at best (Princeton thesis).

**Design consequence:** the spec's "lossless 8×" is impossible for *new* data.
Lossless gains can only come from **redundancy across snapshots**:

1. *Exact dedup*: unchanged vectors are referenced by ID, costing 0 bytes of payload.
2. *XOR delta against the previous version of the same vector ID*: when a vector is
   re-embedded with last-bit noise, `bits(v_new) XOR bits(v_old)` is mostly zero
   in the sign/exponent/high-mantissa bits. After **byte-plane shuffling** (all byte-0s
   together, all byte-1s together …) zstd crushes the zero planes. This is the
   Gorilla idea applied *across time per ID* rather than *across neighbours*.
3. *New vectors*: byte-shuffle + zstd only (≈1.1–1.2×, honestly reported).

## 4. Lossy "semantic mode" — Residual / Product Quantization

* **PQ** (Jégou, Douze, Schmid, TPAMI 2011): split a D-dim vector into M sub-vectors,
  run k-means (k=256) per subspace, store one byte per subspace → D·4/M compression.
* **IVF-ADC / residual PQ**: first assign to a coarse centroid c*, then PQ-encode
  r = v − c*. Residuals have lower variance, so PQ error drops. This is *exactly*
  what the spec calls RQ-Dedup, and it is what Faiss `IndexIVFPQ` does.
* Recall loss is **not mathematically bounded**. At 16–32× compression PQ typically
  loses several % of Recall@10 without re-ranking. We therefore measure the
  **compression ↔ recall curve** and let the user pick an operating point.

**Novelty claim we can defend:** not PQ itself (that's 2011), but using it *inside a
versioned, content-addressed snapshot store*: centroids/codebooks are shared objects
deduplicated across snapshots, unchanged vectors cost nothing, and lossy mode is a
per-snapshot policy with a measured recall guarantee.

## 5. HNSW structure (what the graph diff engine has to understand)

Malkov & Yashunin, arXiv:1603.09320 (TPAMI 2018).

* Each node gets a top level `l = floor(-ln(U(0,1)) · mL)`, with `mL = 1/ln(M)`.
* Layer 0 keeps up to `M0 = 2M` neighbours; upper layers keep up to `M`.
* Insert: greedy descent from the entry point through the upper layers, then
  `efConstruction`-wide beam search on each layer ≤ l, then pick neighbours with the
  **diversity heuristic** (keep candidate e only if it's closer to the query than to
  every already-selected neighbour), then add reverse edges and prune overflowing
  neighbour lists with the same heuristic.
* Consequence for snapshots: inserting one node mutates **its own list plus up to
  ~M reverse lists** on every layer it lives on. Inserting x% new nodes touches
  roughly x·(1+M)% of lists, but each touched list changes only by one or two edges.
  That is exactly why **edge-level deltas** beat both full copies and block-level CoW.

**Design consequence:** we implement our own deterministic HNSW in Go (insert,
update, delete-with-repair, search). It gives us direct access to the adjacency lists, and
hnswlib has no Windows wheels for Python 3.12. The graph diff engine works on
the canonical adjacency representation `layer → node → sorted neighbour list`.

## 6. Existing systems

* **LanceDB/Lance**: versions every table change with a new manifest (MVCC, columnar
  fragments). Unchanged *fragments* are shared, but re-embedded rows rewrite whole
  fragments, and the HNSW/IVF index is rebuilt or stored per version. No
  vector-aware delta encoding.
* **Milvus/Qdrant**: snapshots copy segment files; backup tools copy whole segments.
* **Git / Git-LFS / DVC / lakeFS**: content-addressed snapshots of *files*. Excellent
  branching semantics, zero awareness of vector contents.
* We found **no published system that does per-vector delta + HNSW edge-delta
  snapshotting**. This is the genuine gap.

## 7. Storage engine choice

* Spec says RocksDB. RocksDB from Go requires cgo + a C++ build of RocksDB, which is
  painful on Windows. **Pebble** (Cockroach Labs) is a pure-Go LSM engine,
  RocksDB-inspired and format-compatible with RocksDB 6.2.1 for the features CockroachDB uses,
  with atomic batches and a WAL. Same concepts (LSM, memtable, SSTables, atomic
  `WriteBatch`), so it's a fair stand-in and easy to explain.
* Compression: **zstd** via `klauspost/compress` (pure Go).
* Hashing: SHA-256 (hardware-accelerated in Go's stdlib).

## 8. Datasets

* **SIFT1M** (128-d, TEXMEX/IRISA). Hugging Face mirrors: `qbo-odp/sift1m`,
  `hhy3/ann-datasets` (`.fvecs`). The loader supports `.fvecs`.
* **Synthetic "embedding-like" data**: Gaussian-mixture clusters on the unit sphere
  (384/768-d), so the generator is deterministic and needs no download. Churn
  scenarios are generated from it (Section 2 table).

## 9. Revised claims (what we will actually defend)

| Spec claim | Revised, defensible claim (measured) |
|---|---|
| FastCDC = 0% dedup | FastCDC gets 9.7× per snapshot at 1% in-place churn, falls to ~1× at 25%, and is 1.0× on re-embed noise and compaction. VecDedup lossless: 94.8× / 4.0× / 4.6× / 1681×. |
| > 4× on incremental updates | **Met** for lossless up to 25% churn (4.0–94.8× per snapshot, synthetic 384-d; 4.0–86× on SIFT). |
| Lossless 8–16× | Not possible for *new* vectors: we measure 1.16–1.25×, in line with the literature ceiling of ~1.2–1.5×. The big ratios come from cross-snapshot redundancy. |
| ≤0.5% recall loss, "mathematically bounded" | Not bounded. Measured curve: 7.1× → R@10 0.964, 12.7× → 0.885, 21× → 0.742, 32× → 0.603 (384-d). A ≤0.5% loss needs a higher-rate code; reranking with original vectors would be future work. |
| HNSW snapshots by edge delta | **Met**: 1% churn on a 50k SIFT index adds 0.3 MiB vs 10.3 MiB for FastCDC (34×); a restored index answers queries identically. |
| < 200 ms commit on 10M | 1M vectors with 1000 random edits: 327 ms incremental (1.37 s full-scan). Cost ∝ touched segments, not N; 10M not measured on this laptop. |
| < 50 ms branch | **Met**: 0.8–2 ms (one atomic Pebble write). |
