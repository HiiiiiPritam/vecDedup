# VecDedup: Complete Study Guide

> **Note for an AI tutor reading this file:** this document describes a capstone project
> called **VecDedup** that has been fully implemented, tested and benchmarked. The student
> uses it to learn the project from scratch. When answering, explain concepts in a
> technical but beginner-friendly way, build from basics upward, use the concrete numbers
> and file names given here, and do not invent features that are not listed.
> Section 16 lists what is NOT implemented.

---

## Table of contents

1. What the project is, in one paragraph
2. Background concepts (vectors, vector databases, HNSW, snapshots, dedup)
3. The problem being solved
4. Churn: what it is and the types we test
5. The baseline: byte-level deduplication (FastCDC, fixed blocks)
6. Why byte-level dedup fails on vector data (with the maths)
7. What VecDedup does: the core ideas
8. Architecture and code layout
9. Component deep-dives with the maths
   - 9.1 float32 bit layout, XOR delta, byte shuffle, zstd
   - 9.2 Hashing, content addressing, Merkle trees
   - 9.3 Segments and their encodings
   - 9.4 The storage layer (CAS packs + Pebble) and crash safety
   - 9.5 Commit paths, checkout, diff, verify, branches
   - 9.6 HNSW: how it is built, searched, deleted from, and restored
   - 9.7 TGIS: graph edge deltas
   - 9.8 Lossy mode: k-means, product quantization, drift detection
   - 9.9 FastCDC internals (the baseline)
10. How to run it (CLI and benchmarks)
11. How the evaluation was done (methodology)
12. Results, with interpretation
13. Original spec claims vs what we can actually defend
14. What is novel (and what is not)
15. Bugs found during development and how they were fixed
16. Limitations and future work
17. Team split and presentation split
18. Likely viva questions, with answers
19. Glossary
20. References

---

## 1. What the project is, in one paragraph

**VecDedup is a version-control system (like Git) for vector databases.** It stores many
snapshots (versions) of a set of embedding vectors *and* the HNSW search index built on
them, while storing only what actually changed between versions. Regular dedup tools
treat data as raw bytes. VecDedup understands the structure of the data: vectors are
`(ID, float[D])` rows and the index is a graph of `node → neighbour list`. It stores:
- unchanged vectors for free,
- slightly changed vectors as **XOR bit-deltas**,
- index changes as **added and removed edges**.

It supports commits, branches, log, diff, checkout, verify and an optional **lossy mode**
based on product quantization. It is written in Go, and every number in this guide was
measured on our implementation.

---

## 2. Background concepts

### 2.1 Vectors (embeddings)
An **embedding model** (for example `all-MiniLM-L6-v2`, or OpenAI's embedding models) maps a piece of text,
image or audio to a fixed-length array of numbers called a **vector** or **embedding**.

```
"How do I reset my password?"  →  [0.021, -0.113, 0.087, ..., 0.004]   (384 numbers)
```

- **Dimension D** is the array length. Typical values are 128, 384, 768 and 1536.
- Each number is a **float32**, which takes 4 bytes. One vector = `4·D` bytes, so 384-d is 1,536 bytes.
- Texts with similar meanings produce vectors that are close together.

**Distance.** The usual measure is squared Euclidean (L2) distance:

```
dist(a, b) = Σᵢ (aᵢ − bᵢ)²        smaller = more similar
```

For **unit-length** vectors (‖v‖ = 1), ranking by L2 distance is the same as ranking by
cosine similarity, because `‖a−b‖² = 2 − 2·cos(a,b)`. Our code uses L2² everywhere.

### 2.2 Vector databases
A vector database (Milvus, Qdrant, Weaviate, LanceDB, pgvector, …) stores:
1. **the vectors**: N rows of `(id, float[D])`. On disk this is basically an N×D float matrix.
   1M × 384-d is 1.5 GB, and 10M × 768-d is 30 GB.
2. **an index** that makes nearest-neighbour search fast (usually HNSW).

They are the core of **RAG** (Retrieval-Augmented Generation). The user's question is embedded,
the database finds the nearest document vectors, and those documents are given to an LLM.

### 2.3 Nearest-neighbour search and HNSW
**Brute force** compares the query with all N vectors, which costs O(N·D) per query (384M
multiply-adds for 1M × 384). That is too slow at scale, so databases use **approximate nearest
neighbour (ANN)** indexes.

**HNSW** (Hierarchical Navigable Small World, Malkov & Yashunin 2016) is the most popular one:
- It is a **graph**. Each vector is a node, and each node stores an **adjacency list**: the IDs of up to M nearby nodes
  (M = 16 typically; the bottom layer allows 2M = 32).
- It has **layers**. Every node is on layer 0, and a node is also on higher layers with
  exponentially decreasing probability. Upper layers are sparse "express lanes".
- **Search**: start at the entry point on the top layer and greedily move to the
  neighbour closest to the query until nothing is closer, then drop one layer. On layer 0, run a
  **beam search** that keeps the best `ef` candidates. The cost is roughly logarithmic in N.
- **Insert**: find the new node's nearest nodes and link to M of them, then **add the new
  node into each of those neighbours' lists** (links are bidirectional). If a list overflows,
  prune it. **So one insert modifies about M + 1 adjacency lists.** This fact is central to the project.

Index size is about `N × 2M × 4 bytes` for layer 0. For 1M nodes that is about 128 MB, plus the vectors
if they are stored inside the index file (hnswlib does that).

### 2.4 Snapshots and versioning
A **snapshot** is the full state of the data at one moment. Teams want snapshots of their
vector DB for:
- **Rollback**: a bad ingestion job hurt search quality, so go back to yesterday.
- **Reproducibility / audit**: show which data produced a given answer on a given date.
- **Branching / experiments**: copy production, try a new embedding model, compare.

A **commit** creates a snapshot, a **branch** is a named pointer to a snapshot, and the **log**
is the history. This is the same vocabulary as Git.

### 2.5 Deduplication
- **Hash**: a function (we use SHA-256) mapping any bytes to a 32-byte fingerprint. The same input
  always gives the same hash, and collisions are practically impossible.
- **Dedup**: split data into pieces and hash each one. If the hash is already stored, store a reference
  instead of the bytes.
- **Dedup ratio** = logical bytes (what full copies would cost) ÷ physical bytes (what was
  actually written). 10× means 10 times less storage.
- **Incremental ratio** (we report this a lot) is the ratio computed only over the snapshots *after* the
  first one, i.e. "how much does each new snapshot cost compared with a full copy".

---

## 3. The problem being solved

Vector databases change every day (new documents, edits, deletions, re-embedding), and
teams want to keep every version. Copying everything per snapshot is wasteful:
daily snapshots of 30 GB for 30 days come to 900 GB, mostly identical data.

The natural fix, generic byte-level dedup (backup tools, ZFS, Git-LFS-style chunking),
**performs badly on vector data and HNSW indexes**, for structural reasons (Section 6).

**Goal of VecDedup:** snapshot a vector dataset plus its HNSW index so that a new
snapshot costs space roughly proportional to *what semantically changed*. It should also
support fast commits, near-instant branches, exact restore, diff and integrity checking.

---

## 4. Churn

**Churn = the fraction of vectors that changed between two consecutive snapshots.**

```
churn p = (#added + #modified + #deleted) / N
```

With N = 100,000, 1% churn means 1,000 vectors changed and 99,000 are bit-identical to
the previous snapshot.

The *type* of change matters, because each type alters the bytes differently. Our churn
simulator (`internal/synth`) implements these:

| Scenario | What changes in the data | Real-world cause |
|---|---|---|
| `edit-p%` | p% of rows get completely new vectors; IDs and row order are kept | documents edited and re-embedded |
| append | new rows with new IDs at the end | new documents |
| delete | rows removed; later rows shift to new byte offsets | documents removed |
| `mixed-5%` | 50% edit / 30% append / 20% delete of the 5% | a normal day of ingestion |
| `reembed-noise` | **every** vector gets ~1e-6 relative noise, norm preserved | same model re-run on different GPU / batch size |
| `compaction-shuffle` | same rows, different order; no value changes | DB compaction / re-sorting |
| `new-model` | every vector replaced by unrelated new values | embedding-model upgrade (the worst case) |

**Why re-embedding gives different bits:** floating-point addition is not associative
(`(a+b)+c ≠ a+(b+c)` in the last bit). GPUs pick different reduction orders depending on batch
shape and hardware, so embedding the same text twice can differ in the last few bits. This
is documented in the literature (arXiv:2506.09501; Valori arXiv:2512.22280) and in
community reports (Milvus FAQ, OpenAI forum).

---

## 5. The baseline: byte-level deduplication

We compare against what existing tools do (implemented in `internal/cdc`):

- **full-copy**: store every snapshot in full.
- **zstd**: compress each snapshot independently (no cross-snapshot dedup).
- **fixed128K+zstd**: split into fixed 128 KB blocks, dedup by hash, compress. This is what
  ZFS dedup / block-level copy-on-write snapshots effectively do.
- **fastcdc8K+zstd**: content-defined chunking with FastCDC (average 8 KB chunks), dedup by hash,
  compress. This is what restic/Borg-style backup tools do. We count 40 bytes of index overhead per chunk.

**Why content-defined instead of fixed blocks?** With fixed blocks, inserting one byte at the
start shifts every block boundary, so every block hash changes. CDC chooses cut points from the
content (a rolling hash), so boundaries move with the data and most chunks survive insertions.
See 9.9 for the internals.

---

## 6. Why byte-level dedup fails on vector data

### 6.1 A chunk spans several rows
An 8 KB chunk holds `8192 / (4·D)` rows: about **5 rows** for 384-d and 16 rows for 128-d.
If any row in a chunk changed, the whole chunk is new. With random churn p and r rows per chunk:

```
P(chunk unchanged) = (1 − p)^r
```

For r = 5:

| churn p | chunks unchanged | chunks re-stored |
|---|---|---|
| 1% | 95% | 5% |
| 10% | 59% | 41% |
| 25% | 24% | 76% |

So at 25% churn you re-store about 76% of the data, even though only 25% changed.

### 6.2 Re-embedding changes every float slightly
Every row differs in its low bits, so every chunk is new and the ratio is about 1.0×. Bytes cannot
express "almost equal".

### 6.3 Reordering / compaction
The same rows in a different order produce different chunk contents, so dedup is about 1.0×, even though
**no vector changed**.

### 6.4 HNSW index files
One insert modifies about M + 1 adjacency lists at **random positions** in the index file. With
1% new nodes, a large fraction of all chunks contain at least one modified byte, so FastCDC
re-stores most of the file. Measured on a 50k SIFT index at 1% churn: FastCDC adds
**10.3 MiB**, VecDedup adds **0.3 MiB**.

**Root cause:** byte tools see opaque bytes. They don't know about IDs, floats, or edges.

---

## 7. What VecDedup does: the core ideas

1. **Key vectors by ID, not by file position.** Rows are grouped into **segments** of 256
   consecutive IDs (`segment = id >> 8`). Each segment has a **canonical form** (rows sorted by ID,
   exact float bits) and is addressed by the SHA-256 of that form. An unchanged segment has the same hash,
   so it is stored once. Reordering costs nothing, because the canonical form ignores file order.
2. **Delta-encode changed segments against the parent's version of the same segment.**
   Only the removed IDs, the added IDs (with their data) and the changed IDs (with `XOR(new_bits, old_bits)`) are stored.
   Unchanged rows inside a changed segment cost nothing.
3. **Byte-plane shuffle + zstd.** XOR deltas of nearly equal floats are mostly zero
   bytes. Shuffling groups those zeros together and zstd crushes them.
4. **Graph edge deltas (TGIS).** The HNSW index is stored as per-node, per-layer
   adjacency lists, grouped into 256-node segments. A changed segment stores only edges added (ΔE⁺) and
   removed (ΔE⁻).
5. **Merkle tree snapshots.** A snapshot is a tree of hashes (snapshot → trees → segments).
   Shared subtrees are shared physically. A branch is a name pointing to a snapshot hash, so it is O(1)
   to create.
6. **Optional lossy mode.** Vectors are stored as residual product-quantization codes (2 + M
   bytes each). A shared codebook object is reused across snapshots, and the codebook is retrained automatically when the data drifts.
7. **Crash-safe storage.** Objects go to append-only pack files with CRC32. Metadata and refs
   go to Pebble (a pure-Go LSM key-value store, RocksDB-like). Each commit is one atomic batch.

---

## 8. Architecture and code layout

```
 CLI (cmd/vecdedup) ──► repo (commit, branch, checkout, diff, verify, stats)
                          │
          ┌───────────────┼────────────────────────┐
          ▼               ▼                        ▼
   segment codecs     quant (k-means, PQ)     hnsw (graph, dirty tracking)
   VecSeg / CodeSeg / GraphSeg, full | delta
          │
          ▼
   cas: content-addressed object store
   packs/*.vdp (append-only, CRC32)  +  meta/ (Pebble LSM: object index + refs)
```

Language: **Go** (module `vecdedup`). Dependencies: `cockroachdb/pebble` (KV store),
`klauspost/compress` (zstd), `cespare/xxhash/v2` (64-bit fingerprints).

| Package | What it does |
|---|---|
| `internal/vec` | `Dataset{Dim, IDs, Data}` type; `.vdv` file format; `.fvecs` reader (SIFT format); L2/dot/normalize kernels |
| `internal/synth` | deterministic embedding-like data generator + churn simulator (Section 4) |
| `internal/codec` | byte-plane shuffle, XOR of floats, zstd, varint reader/writer with corruption checks |
| `internal/cas` | content-addressed store: pack files + Pebble index, transactions, atomic commit |
| `internal/segment` | segment canonical hashing; full/delta encodings for vectors, PQ codes and graph |
| `internal/repo` | snapshots (Merkle DAG), branches, commit / incremental commit / checkout / diff / verify / stats |
| `internal/hnsw` | our own deterministic HNSW: insert, batch delete with repair, search, dirty tracking, restore |
| `internal/quant` | spherical k-means, residual PQ codebook, encode/decode, drift detection |
| `internal/cdc` | FastCDC and fixed-block dedup baselines |
| `internal/metrics` | brute-force ground truth, Recall@k |
| `cmd/vecdedup` | the command-line tool |
| `cmd/vdbench` | the benchmark harness that produced all results |
| `scripts/plot.py` | charts from `results/results.json` |

Docs: `docs/RESEARCH.md` (literature), `docs/DESIGN.md` (design), `results/RESULTS.md` (numbers),
`README.md` (how to run, team split).

---

## 9. Component deep-dives (with the maths)

### 9.1 float32, XOR delta, byte shuffle, zstd (`internal/codec`)

**float32 layout (IEEE 754):** 32 bits = 1 sign bit, 8 exponent bits and 23 mantissa bits.

```
value = (−1)^sign × 1.mantissa × 2^(exponent − 127)
```

Example:

```
0.12345678 → 0 01111011 11111001101011011101001   (hex 3DFCD6E9)
0.12345679 → 0 01111011 11111001101011011101010   (hex 3DFCD6EA)
```

**XOR** gives 1 only where bits differ:

```
3DFCD6E9 XOR 3DFCD6EA = 00000003   → 30 of 32 bits are zero
```

XOR is reversible: `old XOR delta = new`. So we can store `delta` and rebuild `new` exactly.

**Why new vectors barely compress (the lossless ceiling).** For real-valued embeddings the
23 mantissa bits are close to random (maximum entropy). Only the sign and exponent bits are
predictable. The best published lossless methods reach about **1.2×** (byte-shuffle + entropy coding, ZipNN-style)
to about **1.5×** (Jina's spherical-coordinate transform, arXiv:2602.00079). So "8–16× lossless" on new
data is impossible. Large ratios can only come from **redundancy across snapshots**.

**Byte-plane shuffle.** Each float is 4 bytes (little-endian: byte 0 = lowest mantissa bits,
byte 3 = sign + high exponent). For an array of n floats we write all byte-0s, then all byte-1s,
then all byte-2s, then all byte-3s. After an XOR delta of nearly equal floats, planes 1–3 are almost
all zeros, and placing them contiguously creates long runs that **zstd** (a fast LZ77 + entropy
coder) compresses to almost nothing. For new data, plane 3 (exponents) compresses well and
planes 0–1 don't. That is where the ~1.2× comes from.

**Varints.** IDs and edges are stored as sorted lists using **delta + varint** encoding.
We store the differences between consecutive sorted IDs, each in 1 byte if < 128, 2 bytes if < 16384, and so on.
Consecutive IDs (difference 1) cost 1 byte each.

### 9.2 Hashing, content addressing, Merkle trees

- **Content addressing**: an object's name *is* its SHA-256 hash. Storing the same content
  twice automatically dedups, because the name already exists.
- **Canonical form**: we hash a *canonical* representation (sorted IDs + exact float bits, or sorted
  adjacency lists), **not** the encoded payload. So the address depends only on the content, not on
  whether it was stored as full or delta, and not on history. Two branches that independently
  reach the same state share storage.
- **Merkle tree**: each internal node's hash covers its children's hashes. In VecDedup:

```
Snapshot (hash = Merkle root)
 ├── parents, message, time, dim, count, mode, codebook hash
 ├── VecTrees:   [tree hash, ...]      each tree = 256 segment refs (= 65,536 IDs)
 │                 └── segment refs: (segment index, segment hash, row count)
 ├── GraphHeader hash  (dim, M, efConstruction, seed, entry point, max level)
 └── GraphTrees: [tree hash, ...]
```

If one vector changes, exactly one segment, one tree and the snapshot get new hashes, and
everything else is shared. **Diff** compares trees first and skips identical subtrees, so it is fast.
**Verify** recomputes hashes to detect corruption.

### 9.3 Segments and their encodings (`internal/segment`)

- **Segment** = all rows with the same `id >> 8` (256 IDs). This is the unit of dedup and delta.
- **Object kinds**: VFull, VDelta (vectors); QFull, QDelta (lossy codes); GFull, GDelta (graph);
  Codebook; Tree; Snapshot; GraphHeader.
- Every payload starts with a header: base hash (32 bytes) + chain depth (uvarint).

**Vector full encoding:**
```
dim | sorted IDs (delta-varint) | zstd(shuffle(floats))
```

**Vector delta encoding** (against the same segment in the parent snapshot): walk both sorted
ID lists in a merge:
```
dim | removed IDs | changed IDs | added IDs
    | zstd(shuffle(XOR(new, old) for changed rows))
    | zstd(shuffle(data of added rows))
```
Unchanged rows are not stored at all.

**Choosing full vs delta.** Compute the delta first. If it is smaller than 25% of the raw size
(`smallDelta = 0.25`), take it immediately. Otherwise also build the full encoding and keep the
smaller one.

**Delta chain cap.** A delta's base may itself be a delta. The chain depth is capped at
**MaxChain = 8**, after which a full object is written, so reading a segment never decodes
more than 8 objects.

**Note on granularity.** With random 1% churn, a 256-ID segment is untouched with probability
0.99²⁵⁶ ≈ 7.6%, so most segments do get a new (delta) object. But each delta only contains the
~2–3 changed rows plus small ID lists. That is why the ratio is still about 95×.

**Code segments (lossy):** store `(id, 64-bit xxhash fingerprint of the original vector, PQ code)`.
The delta stores removed IDs and upserted `(id, fingerprint, code)` rows.

**Graph segments (TGIS):** see 9.7.

### 9.4 Storage layer and crash safety (`internal/cas`)

- **Pack files** (`packs/*.vdp`): append-only files containing object payloads, each with a
  CRC32 checksum. A half-full pack is reused after reopening.
- **Pebble** (`meta/`): a pure-Go LSM-tree key-value store by Cockroach Labs, inspired by
  RocksDB. Keys:
  - `o/<hash>` → location `{pack, offset, length, kind}`
  - `r/branch/<name>` → snapshot hash, plus HEAD
- **Transaction** (`Txn`): `Put` dedups against both committed and staged objects. `Commit`:
  1. append all new objects to packs and **fsync**;
  2. write **one Pebble batch** (index entries + branch ref) with `Sync` (WAL-durable, atomic).
- **Crash safety**: a crash before step 2 leaves only unreferenced bytes in a pack, and the repo
  still points at the previous commit. A crash after step 2 means the commit is fully durable. Tests check that
  an uncommitted transaction is invisible and that corrupted bytes are detected via CRC.
- **Why Pebble, not RocksDB (the spec said RocksDB)?** RocksDB from Go needs cgo plus a C++ build,
  which is painful on Windows. Pebble uses the same concepts (memtable, WAL, SSTables, compaction,
  atomic write batches), is pure Go, and is used in production by CockroachDB.

### 9.5 Repository operations (`internal/repo`)

**Two commit paths, which produce identical results:**
- `Commit(dataset, index)`: **full scan**. Group all rows into segments, hash each, and compare with the
  parent's segment hash. Equal segments are reused; changed ones are delta-encoded. Cost is O(N) hashing
  plus O(changed) encoding.
- `CommitChanges(upserts, deletes, index)`: **incremental**. Only segments containing a changed ID,
  and graph segments containing an HNSW "dirty" node, are loaded and re-encoded. Cost scales
  with the number of touched segments, not N.
- Tests check that **both paths produce the same Merkle root** (lossless and lossy).

**Other operations:**
- `Checkout(ref)`: resolve a snapshot, decode each segment (following its delta chain), rebuild the
  dataset, and optionally restore the HNSW index from stored adjacency (no rebuild needed).
- `Resolve`: accepts `branch`, `branch~N` (N commits back), or a hash prefix.
- `Diff(a, b)`: Merkle-skipping comparison. Reports vectors added/removed/changed and graph
  ΔE⁺ / ΔE⁻ counts.
- `Verify`: re-hash everything reachable and check it against the addresses.
- `CreateBranch / DeleteBranch / Switch / Log / Stats` (dedup ratio = logical / physical).

### 9.6 HNSW (`internal/hnsw`)

We wrote our own HNSW instead of using hnswlib, because we need direct access to adjacency lists,
determinism and dirty tracking (and hnswlib has no Windows wheel for Python 3.12). Defaults:
**M = 16, efConstruction = 100, seed = 42**, distance L2².

**Level assignment** (paper: `l = ⌊−ln(U) · mL⌋`, with `mL = 1/ln M`):
- We compute U from `hash(id, seed)` instead of a random stream, so **the same ID always gets
  the same level** regardless of insertion order. This makes restore exact.
- `P(level ≥ l) = M^(−l)`. With M = 16, 1 in 16 nodes reaches layer 1, 1 in 256 reaches layer 2, and so on.
- Layer 0 keeps up to 2M = 32 neighbours; upper layers keep up to M = 16.

**searchLayer (Algorithm 2 of the paper):** a beam search with width `ef`. It keeps a min-heap of
candidates and a max-heap of the best `ef` results found. It repeatedly expands the closest candidate,
and stops when the closest remaining candidate is farther than the worst result.

**Insert (Algorithm 1):** greedy descent with ef = 1 through the layers above the node's level, then for
each layer ≤ level: searchLayer with efConstruction, select neighbours with the **heuristic**, link
both ways, and prune any list that is over capacity using the heuristic.

**Neighbour-selection heuristic (Algorithm 4):** go through candidates from closest to farthest.
Keep candidate e only if it is closer to the query than to every already-selected neighbour. This
prefers neighbours in *different directions*, which keeps the graph navigable.

**Sorted neighbour lists:** lists are stored sorted by ID, so search results are a pure function of
(vectors, adjacency). The tests check that **a restored index returns identical results**.

**Batch delete with repair** (FreshDiskANN-style consolidation): for each node that pointed at a
deleted node, **keep all surviving edges** and fill only the freed slots. Candidates come from the
deleted nodes' neighbours: heuristic selection first, then top up with the nearest. If the entry
point is deleted, a new one is chosen. Freed internal slots are reused.

**Dirty tracking:** every node whose adjacency changed since the last commit is marked dirty, so
`CommitChanges` re-encodes only those graph segments. `ClearDirty` runs after a commit.

**Other:** `Apply(upserts, deletes)` (an update = delete + reinsert), `Search(q, k, ef)`,
`Restore(header, vectors, adjacency)`, and `MarshalFlat()`, which writes an hnswlib-style flat file
that the byte-level baselines snapshot.

**Validation on real data (SIFT, 20k vectors, 128-d), Recall@10:** 0.936 at ef = 16, 0.978 at ef = 32,
0.992 at ef = 64, 1.000 at ef = 128. That is in line with hnswlib's behaviour.

### 9.7 TGIS: graph edge deltas (`segment.GraphSeg`)

TGIS is the name used in the spec for the graph-delta format.
- The canonical graph segment is, for 256 consecutive node IDs, each node's level and per-layer **sorted
  neighbour ID lists**.
- The **delta vs parent** stores removed nodes, added nodes (full lists), and for each modified node and
  layer, **ΔE⁺ (edges added)** and **ΔE⁻ (edges removed)** as sorted delta-varint lists.
- `DiffGraph` counts NodesAdded / Removed / Modified and EdgesAdded / Removed.

Why this is small: an insert changes about M + 1 lists, but **each touched list changes by only 1–2 edges**.
A byte tool re-stores the whole chunk around each change. TGIS stores just those edges, a few bytes each.

### 9.8 Lossy mode (`internal/quant`)

**k-means:** partition vectors into K clusters by minimising `Σ ‖v − c(v)‖²`. We use **k-means++
seeding** (on a sample of at most 20k vectors), Lloyd iterations, and parallel assignment. **Spherical** k-means
re-normalises centroids, which suits unit-length embeddings.

**Product quantization (PQ)** (Jégou, Douze, Schmid 2011):
split a D-dim vector into M sub-vectors of D/M dims each. For each subspace, train 256
centroids. Store each sub-vector as the 1-byte index of its nearest centroid.

**Residual PQ / IVFADC (what we implement):**
1. Coarse quantizer: K centroids (default **K = 256**), found with spherical k-means. Store the nearest one, c*, as a 2-byte ID.
2. Compute the residual `r = v − c*`, which has smaller variance than v.
3. PQ-encode r with M sub-quantizers (256 centroids each), giving M bytes.

```
code     = [coarse id: 2 bytes][M bytes]        (2 + M bytes per vector)
decode   v̂ = c[coarse] + concat(sub-centroid_1, ..., sub-centroid_M)
raw size = 4·D bytes   →   code compression = 4D / (2 + M)
```

Default M = D/8 (for 384-d, M = 48: 1536 → 50 bytes). The reported compression ratios also count
the per-vector **8-byte fingerprint** and the codebook, so they are lower than 4D/(2+M).
Constraints: D % M == 0, K ≤ 65536, at least 256 training vectors, at most 50k vectors used for training.

**Fingerprints:** for each row we store `xxhash64` of the *original* float bits. On the next commit,
a row whose new fingerprint matches is unchanged, so its old code is reused without re-encoding or keeping the
original vector.

**Codebook as a shared object:** the codebook is a CAS object. Snapshots that use the same codebook
reference one copy.

**Drift detection:** an old codebook can become bad when the data distribution changes (e.g. a new model).
On each lossy commit we measure on a sample of up to 2000 vectors:

```
relMSE    = Σ ‖v − v̂‖² / Σ ‖v‖²          (relative reconstruction error)
relCoarse = Σ ‖v − c*‖² / Σ ‖v‖²          (relative coarse-residual energy)
```

If either exceeds **1.25 ×** its value at training time (`DriftFactor = 1.25`), the codebook is retrained.
The commit stats report `Retrained = true`.

**Recall@k (how lossy accuracy is measured):** ground truth is the exact top-k by brute force on the
**original** vectors. The system result is the exact top-k over the **reconstructed** vectors.

```
Recall@k = |returned ∩ true top-k| / k
```

**Important:** recall loss is **not mathematically bounded**. It is a measured trade-off curve.

### 9.9 FastCDC internals (the baseline, `internal/cdc`)

- **Gear rolling hash:** `fp = (fp << 1) + G[byte]`, where G is a table of 256 random 64-bit values.
  That is one shift, one add and one lookup per byte, which is 3–12× faster than Rabin fingerprints.
  Because of the shift, the influence of a byte disappears after 64 bytes, so the hash depends on a
  sliding window.
- **Cut test:** cut a chunk when `fp & mask == 0`. A mask with n one-bits gives an expected
  chunk size of 2ⁿ. The bits are spread over the high bits ("zero padding") to widen the effective window.
- **Normalized chunking:** before the average size use a stricter mask **MaskS** (n+1 bits, cut less often);
  after it, a looser **MaskL** (n−1 bits, cut more often). This pulls sizes toward the average.
- **Sub-minimum skipping:** no cut test before the minimum size.
- **Our parameters:** min 2 KB, average 8 KB (n = 13), max 64 KB.

---

## 10. How to run it

Requirements: Go ≥ 1.22. Python 3 + matplotlib only for plots.

```powershell
go test ./...                                   # full test suite (~30 s)
go build -o bin/vecdedup.exe ./cmd/vecdedup
go build -o bin/vdbench.exe  ./cmd/vdbench
```

**CLI demo:**
```powershell
bin/vecdedup gen   --n 20000 --dim 128 --out v1.vdv                 # synthetic data
bin/vecdedup churn --in v1.vdv --op mixed --frac 0.05 --out v2.vdv  # simulate 5% churn
bin/vecdedup gen   --n 100 --dim 128 --seed 9 --out q.vdv           # queries
bin/vecdedup -C demo commit v1.vdv -m "base" --hnsw                 # first snapshot + index
bin/vecdedup -C demo commit v2.vdv -m "5% churn" --hnsw             # index updated incrementally
bin/vecdedup -C demo branch experiment main~1                       # zero-copy branch
bin/vecdedup -C demo log
bin/vecdedup -C demo diff main~1 main                               # vectors + ΔE+/ΔE−
bin/vecdedup -C demo search main --query q.vdv                      # uses restored index
bin/vecdedup -C demo verify
bin/vecdedup -C demo stats                                          # dedup ratio
```

Commands: `init, commit, log, branch, switch, checkout -o, diff, search, verify, stats, gen, churn`.
Churn ops: `edit, append, delete, mixed, reembed, shuffle, newmodel`.
Lossy: `commit data.vdv --lossy --pq-m 48 [--k 256] [--retrain]`. HNSW parameters: `-M`, `--efc`.

**File format `.vdv`:** `"VDV1" | dim u32 | n u64 | for each row: id u64 + D floats`. `.fvecs` (SIFT) is also read.

**Benchmarks:**
```powershell
# optional real data: first 100k SIFT1M vectors (HTTP range request, 52 MB)
curl.exe -L -r 0-51599999 -o data/sift/sift_base_100k.fvecs https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_base.fvecs
curl.exe -L -o data/sift/sift_query.fvecs https://huggingface.co/datasets/qbo-odp/sift1m/resolve/main/sift_query.fvecs
bin/vdbench --quick        # ~1 min smoke run
bin/vdbench                # full run (~20–25 min) → results/results.json + results/RESULTS.md
python scripts/plot.py     # charts into results/*.png
```
`vdbench` flags: `--quick`, `--only <section>` (merges into the existing results.json), `--sift <dir>`, `--out <dir>`.

---

## 11. Evaluation methodology

**Datasets:**
- **Synthetic 50k × 384-d**: a deterministic "embedding-like" generator. Each of several topic clusters has
  its own random low-rank subspace (16 latent directions), plus a small isotropic noise floor,
  normalised to unit length. Real embeddings have low intrinsic dimension. Pure isotropic Gaussian noise
  would be unrealistically hard for ANN search.
- **SIFT 100k × 128-d** (real data, the first 100k of SIFT1M, a standard ANN benchmark) and SIFT 50k for the
  HNSW experiment.

**Vector experiment:** for each of the 8 churn scenarios, take a base snapshot plus 3 churned snapshots.
Feed all 4 snapshots to every method (full-copy, zstd, fixed128K+zstd, fastcdc8K+zstd,
vecdedup-lossless, vecdedup-lossy with M = D/8). Report the **cumulative ratio** (all 4 snapshots) and the
**incremental ratio** (snapshots 2–4 only). Lossy recall uses in-distribution queries (blends of
dataset vectors).

**HNSW experiment:** build HNSW on SIFT 50k, apply 1/5/10% mixed churn *incrementally*, and snapshot.
Measure bytes added by each method (byte baselines snapshot the hnswlib-style flat file), plus ΔE⁺/ΔE⁻
and Recall@10 before, after and after restore.

**Lossy curve:** M = D/16, D/8, D/4, D/2. Report bytes/vector, compression, Recall@10 and Recall@100.

**Latency:** 100k and 1M vectors at 128-d, 1000 changed vectors. Measure initial commit, full-scan
commit, incremental commit, branch creation and checkout.

Environment: Windows 11, amd64, 16 CPUs, Go 1.26.

---

## 12. Results

### 12.1 Vector snapshots: incremental dedup ratio (higher is better)

**Synthetic 50k × 384:**

| Scenario | FastCDC 8K | VecDedup lossless | VecDedup lossy (M=D/8) |
|---|---|---|---|
| edit 1% | 9.7× | **94.8×** | 1063× |
| edit 5% | 2.4× | **19.8×** | 398× |
| edit 10% | 1.5× | **10.0×** | 226× |
| edit 25% | 1.1× | **4.0×** | 98× |
| mixed 5% | 3.1× | **26.3×** | 470× |
| re-embed noise | 1.0× | **4.6×** | 26× |
| compaction/shuffle | 1.0× | **1681×** | 1741× |
| new model | 1.0× | 1.2× | 21× |

**SIFT 100k × 128 (lossless):** edit 1/5/10/25% = 86.2 / 19.5 / 9.9 / 4.0×; mixed 24.6×;
re-embed 3.3×; compaction 577×; new model 1.0×. FastCDC: 11.5 / 3.4 / 2.4 / 1.7 / 4.2 / 1.8 / 3.1 / 1.0×.
(SIFT bytes are small integers stored as floats, so even plain zstd gets about 3×.)

**How to read this:**
- Lossless VecDedup's incremental ratio is roughly 1/churn (1% gives ~95×, 25% gives 4×), because it stores
  only the changed rows. FastCDC degrades much faster, as explained by `(1−p)^r`.
- **Compaction** is free for VecDedup because segments are keyed by ID.
- **Re-embed** 4.6× comes purely from XOR deltas of near-identical floats.
- **New model** is ~1× for everyone lossless. Nothing can dedup genuinely new data, and 1.2× is the
  entropy ceiling.

### 12.2 HNSW index snapshots (SIFT 50k, bytes added by one snapshot)

| Churn | Touched nodes | ΔE⁺ | ΔE⁻ | Full copy | FastCDC+zstd | **VecDedup** | Recall@10 before / after / restored |
|---|---|---|---|---|---|---|---|
| 1% | 19% | 15,837 | 12,146 | 31.5 MiB | 10.3 MiB | **301.5 KiB** | 0.990 / 0.995 / 0.995 |
| 5% | 59% | 77,316 | 58,858 | 31.7 MiB | 11.3 MiB | **1020 KiB** | 0.990 / 0.993 / 0.993 |
| 10% | 81% | 149,213 | 112,165 | 31.8 MiB | 11.8 MiB | **1.8 MiB** | 0.990 / 0.992 / 0.992 |

Note that 1% churn touches **19% of nodes** (each change affects about M other lists), which is exactly why
byte tools fail. VecDedup is about 34× smaller than FastCDC at 1%, and the restored index has the same recall.

### 12.3 Lossy mode: compression vs recall

| Dataset | M | Bytes/vector | Compression | Recall@10 | Recall@100 |
|---|---|---|---|---|---|
| synthetic 384 | 24 | 26 | 31.7× | 0.603 | 0.730 |
| synthetic 384 | 48 | 50 | 21.2× | 0.742 | 0.831 |
| synthetic 384 | 96 | 98 | 12.7× | 0.885 | 0.935 |
| synthetic 384 | 192 | 194 | 7.1× | 0.964 | 0.984 |
| SIFT 128 | 8 | 10 | 24.4× | 0.432 | 0.549 |
| SIFT 128 | 16 | 18 | 17.7× | 0.632 | 0.709 |
| SIFT 128 | 32 | 34 | 11.5× | 0.766 | 0.832 |
| SIFT 128 | 64 | 66 | 6.6× | 0.910 | 0.942 |

Higher compression gives lower recall. This is a trade-off, not a guarantee. The usual way to recover recall
is re-ranking with exact vectors, which is future work.

### 12.4 Latency

| Vectors | Initial commit | Full-scan commit | Incremental commit (1000 changes) | Branch | Checkout |
|---|---|---|---|---|---|
| 100k × 128 | 311 ms | 133 ms | **96 ms** | **2.1 ms** | 45 ms |
| 1M × 128 | 4030 ms | 1366 ms | **327 ms** | **0.8 ms** | 898 ms |

### 12.5 Tests
`go test ./...` passes all packages. Tests cover:
- bit-exact history,
- an unchanged commit costs ~0 bytes,
- incremental root == full-scan root (both modes),
- graph restore + incremental commit + diff,
- branches,
- lossy recall,
- drift retraining (including a 0.01× rescale case),
- reopening the repo,
- bad input,
- CAS persistence, uncommitted transactions being invisible, and corruption detection,
- HNSW recall, delete repair, entry-point deletion, identical results after restore, and dirty tracking,
- codec round-trips and CDC behaviour.

---

## 13. Original spec claims vs what we can defend

| Original spec claim | Status | Defensible statement |
|---|---|---|
| "FastCDC gets 0% dedup on vectors" | **False in general** | FastCDC gets 9.7× at 1% in-place churn, ~1× at 25%, and 1.0× on re-embed noise, compaction and HNSW updates |
| > 4× dedup on incremental updates | **Met** | lossless 4.0–94.8× per snapshot at 1–25% churn (synthetic), 4.0–86× (SIFT) |
| Lossless 8–16× | **Impossible for new data** | 1.16–1.25× measured; literature ceiling ~1.2–1.5×; large ratios come from cross-snapshot redundancy |
| ≤ 0.5% recall loss, "mathematically bounded" | **Not met / not true** | measured curve: 7.1× gives R@10 0.964, 32× gives 0.603 |
| HNSW snapshots via edge deltas | **Met** | 0.3 MiB vs 10.3 MiB (FastCDC) at 1% churn; identical results after restore |
| < 200 ms commit on 10M vectors | **Not shown** | 327 ms incremental at 1M; cost scales with touched segments; 10M not measured |
| < 50 ms branch | **Met** | 0.8–2 ms |
| RocksDB | **Changed** | Pebble (pure-Go, RocksDB-like), for Windows buildability |
| gRPC server, FUSE mount | **Not built** | listed as future work |

---

## 14. What is novel (and what is not)

**Not new (existing building blocks):**
- content-addressed snapshots and Merkle DAGs (Git),
- content-defined chunking (FastCDC 2016),
- XOR float encoding (Facebook Gorilla 2015, applied there across neighbouring time-series values),
- product quantization / IVFADC (2011),
- HNSW (2016),
- LSM key-value stores.

**New (the contribution):** a snapshot store that understands the data and dedups at the level of
**individual vectors and individual graph edges**:
1. per-ID segment dedup that is immune to reordering;
2. XOR deltas **across time for the same vector ID** (Gorilla applies XOR between neighbouring values; we
   apply it between versions);
3. HNSW snapshots as per-node edge deltas (ΔE⁺/ΔE⁻), with a delete-repair that keeps edge churn minimal and
   deterministic levels and sorted lists for exact restore;
4. lossy mode *inside* a versioned store: a shared codebook object, fingerprints to skip unchanged rows,
   delta-encoded code segments, and drift-triggered retraining.

Existing systems: Git / DVC / lakeFS / Git-LFS version files. ZFS dedups blocks. LanceDB shares unchanged
row fragments via versioned manifests but has no per-vector deltas and no index edge deltas. Milvus and Qdrant
backups copy segment files. **In our literature search we found no published system doing per-vector
plus per-edge-delta snapshotting.** Phrase it this way, not "nobody has ever done this".

---

## 15. Bugs found during development and how they were fixed

These are good to mention: they show the benchmarks were used to validate the system.

1. **Delete repair destroyed edges.** The first version re-pruned whole neighbour lists after a delete,
   removing 77k edges and adding only 17k. That hurt recall and made graph deltas large. **Fix:** keep all surviving
   edges and fill only the freed slots. Edge additions and removals became roughly balanced.
2. **Lossy mode missed distribution drift.** After a model change or a rescale, the old codebook was reused and
   recall fell to about 0.01. **Fix:** relative drift metrics (relMSE and relCoarse against training values) with
   a 1.25× threshold, plus automatic retraining.
3. **Lossy incremental commit fingerprinted reconstructed vectors** instead of originals, so rows looked changed
   when they weren't. **Fix:** build the code segment directly from the parent's codes and fingerprints.
4. **Synthetic data was unrealistically hard** for HNSW (isotropic noise gave low recall). **Fix:** a low-rank
   latent cluster generator. The SIFT test confirmed the HNSW itself was correct.
5. **Benchmark harness problems:** the re-embed simulator rescaled SIFT vectors (it now preserves the norm), and
   lossy recall was measured with out-of-distribution queries (it now uses in-distribution queries).
6. **Incremental commit was no faster than a full scan.** **Fix:** smaller segments (shift 10 → 8, i.e. 1024 → 256 IDs)
   and the "delta-first" shortcut in the encoder.

---

## 16. Limitations and future work

- **No garbage collection:** unreachable objects are not deleted after branch deletion. Future work: mark-and-sweep
  over the Merkle DAG plus pack compaction.
- **Single writer** per repository; no cross-process lock.
- **HNSW build is single-threaded.**
- **No gRPC server, no FUSE mount, no O_DIRECT** (these are Linux-specific or out of scope).
- **Incremental commit cost grows with the number of touched segments.** Random edits spread across many
  segments. Segment size (256) is a tunable trade-off.
- **10M scale not measured.**
- **Lossy recall** is below a ≤0.5% loss target at high compression. Future work: exact re-ranking and
  higher-rate codes.
- **Real re-embedding data:** the re-embed scenario is simulated (1e-6 relative noise). Future work:
  embed a real corpus twice on different hardware and measure.
- Extension ideas: spherical-coordinate transform (Jina 2026) for new vectors, parallel HNSW
  build, an hnswlib file importer, and a ZFS comparison in WSL.

---

## 17. Team split and presentation split

| | Member A: Vector dedup | Member B: Graph engine | Member C: Storage & system |
|---|---|---|---|
| Owns | `codec`, `quant`, vector + code parts of `segment`, `synth` | `hnsw`, graph part of `segment` (TGIS), `metrics` | `cas`, `repo`, `cdc`, CLI, `vdbench`, plots |
| Must explain | float32 layout, byte shuffle, XOR delta, lossless ceiling, k-means, PQ / IVFADC, drift | HNSW algorithms 1/2/4, levels, heuristic, delete-repair, why one insert touches ~M lists, TGIS | content addressing, Merkle trees, Git object model, LSM / Pebble, crash-safe commit, FastCDC |
| Owns results | vector table, lossy curve | HNSW table, SIFT recall | latency, baseline methodology |

**Presentation (~18 min):**
1. Problem (C)
2. Where byte dedup works and fails (C)
3. Architecture (C)
4. Lossless (A)
5. Lossy (A)
6. Vector results (A)
7. HNSW basics (B)
8. TGIS + delete-repair (B)
9. Graph results (B)
10. Live demo (C)
11. Latency + crash safety (C)
12. Limitations (all)

---

## 18. Likely viva questions, with answers

**Q: Isn't the lossy mode just Faiss IVF-PQ?**
A: The quantizer is IVFADC (2011), yes. The contribution is using it inside a versioned content-addressed store:
the codebook is a shared object, fingerprints skip unchanged rows, code segments are delta-encoded, and
drift triggers retraining.

**Q: Why not just use Git-LFS / DVC / lakeFS / ZFS?**
A: They dedup files or blocks. Our results show that one compaction or one HNSW update makes them re-store almost
everything (1.0× on compaction; 10.3 MiB vs 0.3 MiB on a 1% HNSW update).

**Q: Why can't lossless compression reach 8×?**
A: Embedding mantissa bits are close to random, so their entropy is near maximal. Only the sign and exponent compress, which gives
about 1.2–1.5×. Large ratios only come from redundancy across versions.

**Q: Why does XOR help for re-embedded vectors?**
A: Near-equal floats share sign, exponent and high mantissa bits, so their XOR is mostly zeros. Byte-plane shuffling
puts those zeros together and zstd compresses them heavily.

**Q: Why segments of 256 and not per-vector objects?**
A: Per-vector objects would cost a 32-byte hash plus index entry for each 1.5 KB vector and millions of keys. Per-segment
objects amortise the overhead, and the delta format still makes unchanged rows inside a segment cost nothing.
256 was chosen after measuring that 1024 made incremental commits slow.

**Q: How do you guarantee a restored index is identical?**
A: Levels are derived from hash(id, seed), neighbour lists are stored sorted, and search is a pure function of
vectors + adjacency. A test checks identical search results before and after restore.

**Q: What happens if the machine crashes mid-commit?**
A: Packs are fsynced first, then one atomic Pebble batch publishes the index entries and branch ref. A crash
before the batch leaves only unreferenced bytes, and the repo stays at the previous commit.

**Q: Is your synthetic data realistic?**
A: It models clustered, low-intrinsic-dimension, unit-norm data. All key results were also run on real SIFT data.

**Q: Why is there a full-scan commit and an incremental commit?**
A: Full scan works with no change information (you just give it a dataset) and handles reordering. Incremental uses
known upserts/deletes and HNSW dirty nodes to touch only affected segments. Both give the same Merkle root.

**Q: Is the recall loss bounded?**
A: No. PQ error depends on the data. We report a measured curve and let users pick an operating point.

**Q: Why Go and Pebble?**
A: Single static binary, good concurrency, a fast stdlib SHA-256, and Pebble gives RocksDB-style LSM semantics without
cgo, which is hard on Windows.

---

## 19. Glossary

| Term | Meaning |
|---|---|
| Embedding / vector | fixed-length float array representing meaning |
| D (dimension) | number of floats per vector |
| float32 | 4-byte IEEE 754 number: 1 sign, 8 exponent, 23 mantissa bits |
| Vector database | stores vectors + an ANN index |
| ANN | approximate nearest neighbour search |
| HNSW | layered proximity graph for ANN search |
| M | max neighbours per node per upper layer (2M on layer 0) |
| ef / efConstruction | beam width at search / build time |
| Adjacency list | the neighbour IDs of a node |
| Snapshot / commit | saved state at one time |
| Branch | named pointer to a snapshot |
| Churn | fraction of vectors changed between snapshots |
| Dedup ratio | logical bytes / physical bytes |
| Incremental ratio | ratio over snapshots after the first |
| Hash (SHA-256) | 32-byte content fingerprint |
| Content addressing | object name = hash of content |
| Canonical form | a unique representation used for hashing |
| Merkle tree / DAG | tree where parent hashes cover child hashes |
| Segment | 256 consecutive IDs; the unit of dedup |
| Delta encoding | store differences against a base version |
| XOR delta | bitwise difference of two floats |
| Byte-plane shuffle | regroup bytes by position within each float |
| zstd | Zstandard compressor |
| Varint | variable-length integer encoding |
| CDC / FastCDC | content-defined chunking / its fast Gear-hash variant |
| Gear hash | rolling hash `fp = (fp<<1) + G[b]` |
| LSM tree | log-structured merge tree (the RocksDB/Pebble design) |
| WAL | write-ahead log |
| fsync | force data to disk |
| Pack file | append-only file of stored objects |
| TGIS | our graph delta format (ΔE⁺ added / ΔE⁻ removed edges) |
| Dirty node | a node whose adjacency changed since the last commit |
| k-means | clustering minimising squared distance to centroids |
| PQ | product quantization: per-subspace codebooks, 1 byte per subspace |
| IVFADC / residual PQ | coarse centroid + PQ of the residual |
| Codebook | the trained centroids used for encoding |
| Fingerprint | 64-bit xxhash of an original vector |
| Drift | data distribution shift that makes a codebook bad |
| Recall@k | fraction of the true top-k that the system returns |
| SIFT1M | standard 1M × 128-d ANN benchmark dataset |
| Lossless / lossy | bit-exact / approximate storage |

---

## 20. References

- Xia et al., *FastCDC: a Fast and Efficient Content-Defined Chunking Approach for Data Deduplication*, USENIX ATC 2016; extended in IEEE TPDS 2020.
- *A Thorough Investigation of Content-Defined Chunking Algorithms*, arXiv:2409.06066.
- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs*, arXiv:1603.09320 (TPAMI 2018).
- Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, TPAMI 2011.
- Pelkonen et al., *Gorilla: A Fast, Scalable, In-Memory Time Series Database*, VLDB 2015 (XOR float encoding).
- Lossless embedding compression via spherical coordinates (Jina AI), arXiv:2602.00079.
- Floating-point transformation for lossless compression, arXiv:2506.18062.
- Embedding non-determinism: arXiv:2506.09501; Valori, arXiv:2512.22280.
- Pebble: https://github.com/cockroachdb/pebble ; Cockroach Labs blog "Pebble: a RocksDB-inspired key-value store".
- LanceDB / Lance format docs: https://lancedb.github.io/lancedb/
- SIFT1M mirror: https://huggingface.co/datasets/qbo-odp/sift1m
- Project docs: `docs/RESEARCH.md`, `docs/DESIGN.md`, `results/RESULTS.md`, `README.md`.
