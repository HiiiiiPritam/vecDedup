# VecDedup: Presentation Content

Everything needed to build the slides: slide-by-slide text, numbers, formulas, definitions,
visuals, speaker notes, a demo script and a Q&A bank. Every number below comes from
`results/RESULTS.md` or a measured demo run (see `docs/LEARNING_NOTES.md` for where each one
comes from).

**Audience assumption:** capstone evaluators who know basic CS (hashing, graphs, compression)
but not necessarily vector databases. Define terms the first time they appear.

**Time budget:** about 18 minutes plus Q&A, 14 main slides, about 1.3 minutes each.

---

## 0. The story in 30 seconds (open and close with this)

> Vector databases power RAG chatbots, and teams need version history of them: roll back, audit,
> branch. Copying everything per version is too costly, and backup-style chunk dedup breaks on
> vector data. VecDedup is Git for vector databases. It identifies vectors by ID, stores changed
> vectors as bit-level deltas and the HNSW index as edge changes, and shares everything else by
> hash. At 1% daily change a snapshot costs about 95× less than a full copy (FastCDC: about 10×),
> and an index snapshot costs 0.3 MiB instead of 10.3 MiB.

**Three headline numbers:** **94.8×** (vs 9.7×) at 1% churn · **0.3 MiB vs 10.3 MiB** for an index
update · **about 1 ms** to create a branch.

---

## 1. Main slides

Owners follow the team split: **A** = vector dedup and lossy, **B** = graph engine, **C** = storage
and system. Rename as needed.

### Slide 1: Title
**Owner:** C · **Time:** 0:30
- **VecDedup: Semantic-aware snapshotting and deduplication for vector databases**
- Team names, guide, institute, date
- Tagline: *Git-style version control for embeddings and their HNSW index*

---

### Slide 2: The problem
**Owner:** C · **Time:** 1:30
**Key message:** teams need version history of their vector data, and storing full copies doesn't scale.

On-slide:
- RAG systems turn documents into **embeddings** (arrays of float32, e.g. 384 numbers = 1,536 bytes)
- A vector database stores the vectors plus an **HNSW index** for fast search
- 10M documents × 768-d ≈ **30 GB** of vectors, plus several GB of index
- Why versions are needed: **rollback · audit/reproducibility · branching experiments**
- Daily full copies: 30 days × 30 GB ≈ **900 GB**, mostly identical data

Formula/number: `size = N × D × 4 bytes`.
Visual: a simple diagram: *documents → embedding model → vectors → vector DB (vectors + HNSW graph)*, with a
"snapshot per day" stack of boxes growing.
Speaker notes: define embedding and HNSW briefly (one sentence each). Say "version control, not backup".

---

### Slide 3: Why existing dedup fails
**Owner:** C · **Time:** 2:00
**Key message:** byte-level chunk dedup sees opaque bytes, so it breaks on vector data in four ways.

On-slide (four short bullets):
1. **A chunk holds several rows** → one change re-stores the whole chunk
2. **Re-embedding flips the last bits** of every float → every chunk looks new
3. **Compaction reorders rows** → same data, different chunks
4. **One HNSW insert edits about 17 scattered neighbour lists** → most chunks of the index file change

Formula: `P(chunk unchanged) = (1 − p)^r`, with r ≈ 5 rows per 8 KB chunk (384-d):

| Churn p | Chunks unchanged | Re-stored |
|---|---|---|
| 1% | 95% | 5% |
| 10% | 59% | 41% |
| 25% | 24% | **76%** |

Measured, FastCDC incremental ratio: **9.7× (1%) → 1.5× (10%) → 1.1× (25%)**; **1.0×** on re-embed, compaction and index updates.
Visual: chunk diagram: 4 chunks of 5 rows each, with 1 changed row marked in each, all chunks highlighted; next to it
the two floats `3DFCD6E9` vs `3DFCD6EA`.
Speaker notes: explain "chunk = about 8 KB piece, FastCDC picks cut points from content". Mention that the spec's
"FastCDC gets 0%" isn't universally true; we measured where it works and where it fails.

---

### Slide 4: Our idea and architecture
**Owner:** C · **Time:** 2:00
**Key message:** make the storage layer understand vectors, and share everything unchanged by hash.

On-slide:
- Identify vectors by a **stable ID**; group into **segments**: `segment = id >> 8` (256 IDs)
- **Content addressing:** segment address = SHA-256 of its sorted canonical content
- **Snapshot = a recipe of hashes** (Merkle tree): snapshot → trees → segments
- Unchanged segment → same hash → **0 bytes**; changed → **delta vs. parent**; **branch = a name pointing to a snapshot**
- Storage: append-only pack files + **Pebble** (pure-Go LSM, RocksDB-like) with atomic commits

Visual (core diagram, draw it): the Day 1/2/3 example:
```
Day 1: a1 b1 c1 d1 e1 stored          C1 → [a1, b1, c1, d1, e1]
Day 2: b2 (small delta), f1 new       C2 → [a1, b2, c1, d1, e1, f1]
Day 3: b3, d2 (small deltas)          C3 → [a1, b3, c1, d2, e1, f1]
```
Numbers: logical 1,580 KB vs physical 548 KB after 3 snapshots (≈ 2.9×), ≈ 23× after 30.
Speaker notes: stress "a snapshot stores pointers, not vectors". Reordering is free because rows are sorted by ID before hashing.
Assumption to mention: IDs are stable (primary keys).

---

### Slide 5: Lossless vector deltas
**Owner:** A · **Time:** 2:00
**Key message:** a changed segment stores only what changed, and "almost equal" floats become mostly zeros.

On-slide:
- Compare a segment with the parent's, matched by ID: **removed · changed · added**
- Changed rows: **XOR** of new bits with old bits (reversible, bit-exact)
- **Byte-plane shuffle**, then **zstd**
- New vectors: shuffle + zstd only, ceiling about **1.2×**

Visual: the XOR table:
```
old       new       XOR
3DFCD6E9  3DFCD6EA  00000003
BE4CCCCD  BE4CCCCC  00000001
3F19999A  3F19999A  00000000
3C23D70A  3C23D70B  00000001
```
and the shuffle: `03 00 00 00 | 01 00 00 00 …` → `03 01 00 01 | 00 00 00 00 | 00 00 00 00 | 00 00 00 00`.
Numbers (one real segment, 256 × 384): 3 edited rows → delta **4,675 B** vs **340,505 B** full · 3 noisy rows → **1,201 B** ·
whole segment re-embedded: XOR+zstd **118,394 B**, XOR+shuffle+zstd **86,288 B** (4.6× vs 393,216 raw) · new data:
zstd alone 393,238 B, shuffle+zstd 340,181 B (**1.16×**).
Formula: `old XOR delta = new`.
Speaker notes: say clearly "we do not eliminate the floating-point noise; we store it cheaply and exactly". Explain that zstd is
standard; our contribution is making the data compressible.

---

### Slide 6: Lossless results
**Owner:** A · **Time:** 1:30
**Key message:** per-snapshot cost is about proportional to what changed; byte-level tools fall away quickly.

Chart: `results/vectors_synthetic-50000x384.png` (and `results/vectors_sift-100kx128.png` as a second panel or backup).

Table to show (incremental ratio, 50k × 384):

| Scenario | FastCDC | **VecDedup** |
|---|---|---|
| edit 1% | 9.7× | **94.8×** |
| edit 5% | 2.4× | **19.8×** |
| edit 10% | 1.5× | **10.0×** |
| edit 25% | 1.1× | **4.0×** |
| mixed 5% | 3.1× | **26.3×** |
| re-embed noise | 1.0× | **4.6×** |
| compaction | 1.0× | **1681×** |
| new model | 1.0× | 1.2× |

On-slide takeaway: *lossless ratio ≈ 1 / churn; the new-model case is the lossless limit for everyone.*
SIFT (real data, lossless): 86.2 / 19.5 / 9.9 / 4.0× at 1/5/10/25% edits; compaction 576.7×; re-embed 3.3×.
Speaker notes: explain "incremental ratio = full-copy size ÷ bytes the snapshot added". Be upfront that the new-model row is about
1.2×. The compaction scenario also deletes 1%.

---

### Slide 7: Lossy (semantic) mode
**Owner:** A · **Time:** 1:30
**Key message:** trade a little accuracy for 7–30× smaller storage with a shared codebook, and detect when it goes stale.

On-slide:
- **Residual product quantization** (IVFADC): coarse centroid (K = 256) + PQ of the residual
- Code = **2 + M bytes** (default M = D/8 → **50 bytes** instead of 1,536 for 384-d)
- **Shared codebook** object · **64-bit fingerprint** per vector to skip unchanged rows
- **Drift detection:** retrain if error > **1.25×** the training error

Formulas:
```
v̂ = c[coarse] + concat(sub-centroid_1 … sub-centroid_M)
size ratio ≈ N·4D / ( N·(2 + M + 8) + codebook ),   codebook ≈ 2·256·D·4 B ≈ 0.79 MB (D = 384)
relMSE = Σ‖v − v̂‖² / Σ‖v‖²
Recall@k = |returned ∩ true top-k| / k
```
Visual: the 4-number toy: `v → coarse C0 → residual → codes [0,1,0] → decoded`; error **0.0036** (residual PQ) vs **0.0116** (plain PQ).
Speaker notes: PQ itself is from 2011; our contribution is using it inside a versioned content-addressed store.

---

### Slide 8: Lossy results
**Owner:** A · **Time:** 1:00
**Key message:** it's a measured trade-off curve, not a guarantee.

Chart: `results/lossy_tradeoff.png`.

| Data | Compression | Recall@10 |
|---|---|---|
| synthetic 384-d | 31.7× | 0.603 |
| synthetic 384-d | 21.2× | 0.742 |
| synthetic 384-d | 12.7× | 0.885 |
| synthetic 384-d | 7.1× | 0.964 |
| SIFT 128-d | 24.4× / 17.7× / 11.5× / 6.6× | 0.432 / 0.632 / 0.766 / 0.910 |

On-slide callout: *the original "≤ 0.5% recall loss, mathematically bounded" claim is not achievable; we report the real curve.*
Also: lossy snapshots cost about 58 B per changed vector (re-embed 26×, new model 21×).
Speaker notes: mention the drift-detection bug story (stale codebook → recall 0.01 → fixed by retraining).

---

### Slide 9: HNSW in one slide
**Owner:** B · **Time:** 1:30
**Key message:** one insert changes about M + 1 neighbour lists, which is why indexes are hard to snapshot.

On-slide:
- Each vector is a **node** with up to 2M (layer 0) or M (upper layers) neighbour IDs
- **Layers:** `P(level ≥ l) = M^(−l)` (M = 16)
- **Search:** greedy descent through the layers, then beam search of width `ef`
- **Insert:** find nearest nodes → link → **add the new node into each neighbour's list** → prune if full

Visual: the toy graph before/after inserting X, highlighting E, F, G, X as the 4 lists that changed:
```
E layer 1: [B] → [B, X]        F: [E, G] → [E, G, X]
G:         [F, H] → [F, H, X]  X: new, [F, G]
```
Numbers: 1% new nodes touched **19%** of all nodes on SIFT 50k (9% in a 10k-node M = 16 test).
Formulas: `level = ⌊−ln(U) / ln M⌋` · index size ≈ `N × 2M × 4 B` for layer 0.
Speaker notes: explain greedy search with one example walk (start B, query q, B→D→E→F→G). Our HNSW is our own deterministic Go
implementation, validated on SIFT: Recall@10 0.936 / 0.978 / 0.992 / 1.000 at ef = 16 / 32 / 64 / 128.

---

### Slide 10: Index as edge deltas
**Owner:** B · **Time:** 1:30
**Key message:** store only the edges that changed.

On-slide:
- Per node and layer: **ΔE⁺** (edges added) and **ΔE⁻** (edges removed), as sorted difference-coded IDs
- Not XOR: a neighbour list has no fixed positions (removing one element shifts the rest)
- **Dirty-node tracking** (no full comparison) · **deterministic levels** `hash(id, seed)` · **sorted lists** (canonical form)
- **Delete-repair that keeps surviving edges** (the first version destroyed edges: −77k / +17k)
- **Restore without rebuild:** identical search results

Visual: `[12,90,301,4410,7002] → [12,90,301,5120,7002]` gives `ΔE⁺ = [5120], ΔE⁻ = [4410]` (about 6 bytes).
Formula: `ΔE⁺ = new ∖ old · ΔE⁻ = old ∖ new`.
Speaker notes: contrast with slide 5 ("different data, different difference operator"). Mention the real 10k-node run: 953 nodes
touched, ΔE⁺ 1,786 / ΔE⁻ 145, graph snapshot **8,160 B vs 254,256 B** in full (about 31×).

---

### Slide 11: Index results
**Owner:** B · **Time:** 1:00
**Key message:** about 35× less than FastCDC at 1% churn, with the same recall after restore.

Chart: `results/graph_snapshots.png`.

| Churn | Nodes touched | FastCDC+zstd | **VecDedup** | Recall@10 before / after / restored |
|---|---|---|---|---|
| 1% | 19% | 10.3 MiB | **301.5 KiB** | 0.990 / 0.995 / 0.995 |
| 5% | 59% | 11.3 MiB | **1,020 KiB** | 0.990 / 0.993 / 0.993 |
| 10% | 81% | 11.8 MiB | **1.8 MiB** | 0.990 / 0.992 / 0.992 |

On-slide callout: *FastCDC (10.3 MiB) added more than plain zstd of the whole file (9.7 MiB): it saved nothing.*
Dataset line: SIFT 50k × 128-d, a standard 128-d image-descriptor benchmark (real data).
Speaker notes: graph cost grows faster than vector cost with churn (81% of nodes touched at 10%), but is still about 6.6× smaller than FastCDC.

---

### Slide 12: Version control features and live demo
**Owner:** C · **Time:** 2:00
**Key message:** it behaves like Git: commit, branch, diff, verify, checkout.

On-slide:
- **Branch:** one atomic metadata write, zero-copy (**0.8–2 ms**)
- **Diff:** Merkle comparison, skips equal hashes; reports vectors and ΔE⁺/ΔE⁻
- **Verify:** re-decodes every segment and re-hashes; **a flipped bit is detected**
- **Checkout:** exact restore, including the index (no rebuild)

Live demo: see Section 3. Backup screenshot: the `log`, `diff` and `verify` output.
Speaker notes: demo timing about 90 seconds. Mention "no merge yet".

---

### Slide 13: Performance and reliability
**Owner:** C · **Time:** 1:30
**Key message:** commits cost about what changed, and the store is crash-safe.

Chart: `results/latency.png`.

| Vectors | Full-scan commit | **Incremental commit (1,000 changes)** | Branch | Checkout |
|---|---|---|---|---|
| 100k × 128 | 133 ms | **96 ms** | 2.1 ms | 45 ms |
| 1M × 128 | 1,366 ms | **327 ms** | 0.8 ms | 898 ms |

On-slide:
- Write protocol: **fsync packs → one atomic Pebble batch** (objects + branch pointer): a crash leaves the repo at the previous commit
- **CRC32** on every object; SHA-256 content hash on every segment
- Test suite passes: bit-exact history, incremental root == full-scan root, restore equality, drift retraining, corruption detection

Speaker notes: be exact: crash safety is tested by simulation, and 10M vectors were not measured.

---

### Slide 14: Limits, novelty and takeaways
**Owner:** All · **Time:** 1:30
**Key message:** what we built, what's new, and what remains.

On-slide:
- **New:** a snapshot store that dedups per vector and per graph edge, with XOR deltas across time and lossy mode inside a versioned store
  (not new: Git-style content addressing, FastCDC, PQ, HNSW)
- **We corrected the spec with evidence:** FastCDC isn't 0%; lossless 8–16× is impossible for new data; lossy recall isn't bounded
- **Limits:** no merge, no garbage collection, single writer, 10M not measured, re-embed noise simulated, lossy recall below the spec target
- **Future work:** GC, gRPC server, parallel HNSW build, reranking in lossy mode, real re-embedding data, spherical-coordinate transform for new vectors
- Close with the three headline numbers (Section 0)

---

## 2. Backup slides (appendix)

| # | Title | Content |
|---|---|---|
| B1 | Formula sheet | Section 4 below |
| B2 | Definitions | Section 5 below |
| B3 | FastCDC internals | Gear hash `fp = (fp << 1) + G[b]`; cut when `fp & mask == 0`; avg chunk 2ⁿ; normalized chunking (MaskS before the average, MaskL after); min 2 KB / avg 8 KB / max 64 KB |
| B4 | How zstd compresses | repeats → (offset, length) matches; frequent bytes → shorter codes; 1,000 zeros → 16 B; random bytes can't be compressed |
| B5 | Spec vs reality | table in `docs/LEARNING_NOTES.md` Part 6 |
| B6 | Related systems | Git/DVC/lakeFS (whole files); ZFS (fixed blocks); Milvus/Qdrant (copy segment files); LanceDB (shares unchanged row groups; no per-vector or index deltas) |
| B7 | Datasets and method | synthetic 50k × 384 (clustered, low-rank, unit norm); SIFT1M subset 100k × 128; 8 churn scenarios; 4 snapshots per scenario; baselines: full copy, zstd, fixed 128 KB + zstd, FastCDC 8 KB + zstd |
| B8 | Test list | bit-exact history · unchanged commit ≈ free · incremental root = full root · graph restore equal · branches · lossy recall · drift retrain · corruption detected · uncommitted txn invisible |
| B9 | Why not per-vector objects | 1M vectors: per-vector recipe ≈ 40 MB vs 156 KB for 256-ID segments; a 32 B hash is 64% of a 50 B PQ code |
| B10 | Design decisions | Pebble instead of RocksDB (no cgo on Windows); own HNSW (no Windows wheels; need direct access to adjacency); segment size 256 (1024 made incremental commits slow) |

---

## 3. Live demo script (about 90 seconds)

Run from the repo root (`bin/` holds the built executables; build with
`go build -o bin/vecdedup.exe ./cmd/vecdedup`). Use a small dataset so the HNSW build takes 2 seconds.

```powershell
bin/vecdedup gen   --n 5000 --dim 32 --out v1.vdv
bin/vecdedup churn --in v1.vdv --op mixed --frac 0.05 --out v2.vdv
bin/vecdedup -C demo commit v1.vdv -m "day 1" --hnsw
bin/vecdedup -C demo commit v2.vdv -m "day 2: 5% mixed churn" --hnsw
bin/vecdedup -C demo branch experiment main~1
bin/vecdedup -C demo log
bin/vecdedup -C demo diff main~1 main
bin/vecdedup -C demo verify
bin/vecdedup -C demo stats
```

What it printed in our run (timings vary):

```
[main] snapshot d460a2f063cd  (5000 vectors, lossless mode)
  new objects: 14 (755.02 KiB)
[main] snapshot 119f2724e71b  (5025 vectors, lossless mode)
  vector segments: 5 delta   graph segments: 5 delta   edges +8187 -7319
  new objects: 13 (54.80 KiB)                      <- day 2 costs 14x less than day 1
created branch experiment at d460a2f063cd in 5.1ms (zero-copy)
diff: vectors +74 added -49 removed ~125 changed | graph nodes +74 -49 ~3310, ΔE+ 8187 ΔE- 7319
verify: ok on both snapshots
stats: logical 2.07 MiB, physical 815.39 KiB, dedup ratio 2.60x
```

**Optional tamper demo (30 seconds):** copy the repo folder, flip one byte in `packs/pack-000001.vdp`, then
`bin/vecdedup -C demo_bad verify`. Expected: `error: ... cas: checksum mismatch for <hash>`. The untouched copy verifies `ok`.

Fallback if anything fails live: show the screenshot of this output (take it before the presentation).

---

## 4. Formula sheet

| Topic | Formula |
|---|---|
| Vector size | `bytes = D × 4` (384-d → 1,536 B) |
| Squared L2 distance | `d(a,b) = Σᵢ (aᵢ − bᵢ)²`; unit vectors: `‖a−b‖² = 2 − 2·cos(a,b)` |
| float32 value | `(−1)^sign × 1.mantissa × 2^(exponent − 127)` (1 + 8 + 23 bits) |
| Brute-force search cost | `N × D` multiply-adds per query |
| Churn | `p = (added + modified + deleted) / N` |
| Chunk survival | `P(chunk unchanged) = (1 − p)^r`, `r = chunk bytes / row bytes` |
| Segment of an ID | `seg = id >> 8` (256 IDs) |
| Untouched segment (random churn) | `(1 − p)^256` (0.99²⁵⁶ ≈ 7.6%) |
| Dedup ratio | `logical bytes ÷ physical bytes`; incremental = snapshots 2..n only |
| Merkle parent | `H(child₁ ‖ child₂ ‖ …)` |
| XOR delta | `delta = new ⊕ old`; `old ⊕ delta = new` |
| Delta choice | use delta if `size(delta) < 0.25 × raw`, otherwise keep the smaller of delta and full; chain depth ≤ 8 |
| HNSW level | `level = ⌊−ln(U) / ln M⌋`, `P(level ≥ l) = M^(−l)`; `U` from `hash(id, seed)` |
| HNSW capacity | layer 0: `2M`; layers ≥ 1: `M` (M = 16) |
| Edge deltas | `ΔE⁺ = new ∖ old`, `ΔE⁻ = old ∖ new` |
| PQ code | `2 + M` bytes; `v̂ = c[coarse] + concat(sub-centroids)` |
| Lossy size ratio | `N·4D / ( N·(2 + M + 8) + codebook )`, codebook ≈ `2·256·D·4` B |
| Drift | `relMSE = Σ‖v−v̂‖² / Σ‖v‖²`; retrain if `> 1.25 ×` training value |
| Recall | `Recall@k = |returned ∩ true top-k| / k` |
| FastCDC | `fp = (fp << 1) + G[byte]`; cut if `fp & mask == 0`; average chunk `2ⁿ` |

---

## 5. Definitions (copy-paste for slides)

- **Embedding / vector:** a fixed-length array of floats that represents the meaning of a piece of content.
- **Vector database:** stores vectors and an index for similarity search.
- **HNSW:** a layered proximity graph; each node stores its nearest neighbours; search hops toward the query.
- **Snapshot:** the full state of the data at one moment. **Branch:** a name pointing to a snapshot.
- **Churn:** the fraction of vectors that changed between two snapshots.
- **Deduplication:** storing identical content once and referencing it. **Dedup ratio:** logical size ÷ physical size.
- **Content addressing:** naming an object by the hash of its content.
- **Merkle tree:** a tree where each node's hash covers its children's hashes.
- **Segment:** the vectors whose IDs share the same `id >> 8` (up to 256 rows); the unit of dedup.
- **Delta:** the difference from the parent's version of the same object.
- **XOR delta:** the bitwise difference of two floats' bit patterns; mostly zeros when the values are close.
- **Byte-plane shuffle:** regrouping bytes by their position within each float so that zeros sit together.
- **zstd:** a lossless compressor (repeat matching plus entropy coding).
- **ΔE⁺ / ΔE⁻:** edges added to / removed from a node's neighbour list.
- **Product quantization (PQ):** split a vector into M parts and replace each by the 1-byte ID of the nearest table entry.
- **Codebook:** the trained table of centroids. **Fingerprint:** a 64-bit hash of the original vector.
- **Drift:** when the data distribution moves away from what the codebook was trained on.
- **Recall@k:** the fraction of the true k nearest neighbours that the system returns.
- **SIFT1M:** a standard benchmark of 1M 128-d image-descriptor vectors with queries and true neighbours.

---

## 6. Numbers cheat sheet

**Environment:** Windows 11, 16 CPUs, Go 1.26. Datasets: synthetic 50,000 × 384; SIFT 100,000 × 128 (index tests: SIFT 50,000).

| Claim | Number |
|---|---|
| Lossless, 1% edits | **94.8×** (FastCDC 9.7×), SIFT 86.2× |
| Lossless, 25% edits | **4.0×** (FastCDC 1.1×) |
| Re-embed noise | **4.6×** synthetic, 3.3× SIFT (FastCDC 1.0×) |
| Compaction (delete 1% + shuffle) | **1681×** (FastCDC 1.0×); pure shuffle adds 169 bytes |
| New model | 1.2× (lossless ceiling about 1.2–1.5×) |
| New-data compression | 1.16× (shuffle + zstd), zstd alone 1.00× |
| Index snapshot at 1% / 5% / 10% | **301.5 KiB / 1,020 KiB / 1.8 MiB** vs FastCDC 10.3 / 11.3 / 11.8 MiB |
| Nodes touched at 1% / 5% / 10% | 19% / 59% / 81% |
| Lossy compression ↔ recall (384-d) | 31.7× → 0.603 · 21.2× → 0.742 · 12.7× → 0.885 · 7.1× → 0.964 |
| Lossy (SIFT) | 24.4× → 0.432 · 17.7× → 0.632 · 11.5× → 0.766 · 6.6× → 0.910 |
| HNSW recall (SIFT 20k, our own HNSW) | 0.936 / 0.978 / 0.992 / 1.000 at ef = 16 / 32 / 64 / 128 |
| Branch | 0.8–2 ms (CLI demo 5.1 ms) |
| Commit, 1M vectors, 1,000 changes | **327 ms** incremental, 1.37 s full scan |
| Checkout | 45 ms (100k), 898 ms (1M) |
| Demo (5,000 × 32 + index) | day 1: 755 KiB, day 2: 54.8 KiB (5% churn), 2.6× overall |
| Reorder demo (20k × 384) | 79 of 79 segments reused, 169 bytes added |
| Toy example sizes | delta object 134 B vs 168 B full |

**Say these carefully:** the compaction scenario includes deleting 1%; the new-model result is about 1.2× lossless; 10M vectors were
not measured; re-embedding noise is simulated; lossy recall is a measured trade-off, not a guarantee.

---

## 7. Q&A bank

**Isn't lossy mode just Faiss IVF-PQ?** The quantizer is IVFADC (2011). What is new is using it inside a versioned content-addressed store:
a shared codebook object, per-vector fingerprints to skip unchanged rows, delta-encoded code segments and drift-triggered retraining.

**Why not Git-LFS, DVC, lakeFS or ZFS?** They dedup files or blocks. In our results, a compaction or one index update makes block-level
tools store almost everything again (1.0× on compaction, 10.3 MiB vs 0.3 MiB on an index update).

**Why can't lossless compression reach 8×?** The low mantissa bits of embeddings are close to random, so only the sign and exponent compress,
about 1.2–1.5× (we measure 1.16–1.25×). Large ratios only come from redundancy across snapshots.

**Why does XOR help for re-embedded vectors?** Nearly equal floats share sign, exponent and high mantissa bits, so the XOR is mostly zeros.
Byte-plane shuffle puts the zeros together and zstd compresses them. We keep the result bit-exact.

**Did you fix floating-point noise?** No. We store it cheaply and exactly. A tolerance-based mode would save more but would be lossy; not built.

**Why segments of 256 and not per-vector objects?** Per-vector hashes would cost about 40 MB of bookkeeping per 1M vectors, more than the 1% of
changed data; a 32-byte hash is 64% of a 50-byte PQ code. Grouping is for addressing, storage is a delta, so changed segments cost only their changed rows.

**What if IDs change between snapshots?** Dedup degrades: every re-numbered vector looks changed. We assume stable primary keys, as vector
databases provide.

**Why not XOR the graph too?** Neighbour lists have no fixed positions; removing one element shifts the rest. We use set differences (ΔE⁺ / ΔE⁻).

**How do you guarantee the restored index is identical?** Levels come from `hash(id, seed)`, neighbour lists are stored sorted, and search is a
pure function of vectors + adjacency. A test checks identical search results before and after restore.

**What happens on a crash mid-commit?** Packs are fsynced first, then one atomic Pebble batch publishes the objects and branch pointer. A crash
before the batch leaves only unreferenced bytes. (Tested by simulation, not by power failure.)

**Is your synthetic data realistic?** It models clustered, low-intrinsic-dimension, unit-norm data. Key results were repeated on SIFT (real,
but image descriptors, not text embeddings).

**What is the recall guarantee in lossy mode?** None. It's a measured curve (0.60 at 32× to 0.96 at 7×). Re-ranking with original vectors is future work.

**Why Go and Pebble?** Single static binary and good concurrency; Pebble gives RocksDB-style LSM semantics with atomic batches and no cgo, which is hard on Windows.

**What's missing?** Merge, garbage collection, gRPC/FUSE, multi-writer, parallel HNSW build, 10M-vector measurement.

---

## 8. Design tips for the slides

- Use the existing charts (`results/*.png`); they share one validated colour palette, so reuse it in diagrams.
- One idea per slide; put the formula or table on the slide and the explanation in your speech.
- Draw two diagrams yourself: (1) the Day 1/2/3 object-store picture (slide 4), (2) the chunk-spoilage picture (slide 3).
- Keep the numbers in Section 6 as the single source of truth so slides don't contradict each other.
- Rehearse the demo once on the actual presentation machine; keep the screenshot as a fallback.
- Practise saying the three honest limitations without apologising: lossy recall isn't bounded, no merge/GC, 10M not measured.
