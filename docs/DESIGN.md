# VecDedup — Design

This describes what is **implemented**. Where it differs from the original spec, see
`RESEARCH.md` §9 for why.

## 1. Big picture

```
 client / CLI ──► repo (commit, branch, checkout, diff, verify)
                    │
        ┌───────────┼────────────────────────┐
        ▼           ▼                        ▼
   segment codecs  quant (k-means, PQ)     hnsw (graph, dirty tracking)
   VecSeg/CodeSeg/GraphSeg, full|delta
        │
        ▼
   cas: content-addressed objects
   packs/*.vdp (append-only, CRC32)  +  meta/ (Pebble LSM: index + refs)
```

Language: Go. Packages under `internal/`:

| package | responsibility | spec subsystem |
|---|---|---|
| `vec` | dataset type, `.vdv`/`.fvecs` IO, distance kernels | 6.1 ingestion |
| `synth` | embedding-like generator + churn simulator | 7.1 datasets |
| `codec` | byte-plane shuffle, XOR delta, zstd, varints | 6.2 |
| `quant` | spherical k-means, residual product quantizer, drift check | 5.1 / 6.2 RQ-Dedup |
| `hnsw` | deterministic HNSW with real deletes + dirty tracking | 5.2 / B |
| `segment` | segment canonical hashing and full/delta encodings | 5.1, 5.2 |
| `cas` | object store + atomic metadata | 6.4 |
| `repo` | Merkle-DAG snapshots, branches, commit/checkout/diff/verify | 6.3 / C |
| `cdc` | FastCDC + fixed-block baselines | 7 baselines |
| `metrics` | brute-force ground truth, Recall@k | 7.2 |

## 2. Data model

**Segment** = all rows whose `id >> 8` is equal (256 IDs). The unit of dedup.

Each segment has a **canonical form** (sorted IDs + exact float bits, or sorted
adjacency lists), and its SHA-256 is its **content address**. The stored payload is
either:

* **full**: `zstd(byte-shuffle(floats))` + delta-varint IDs, or
* **delta vs. base**: the same ID range in the parent snapshot:
  * vectors: `removed ids`, `changed ids` + `zstd(shuffle(XOR(new, old)))`, `added ids + data`
  * lossy codes: `removed ids`, `upserted (id, fingerprint, code)`
  * graph: `removed nodes`, `added nodes`, per modified node per layer `(ΔE⁺, ΔE⁻)`
    as sorted delta-varint lists (TGIS)

The encoder computes the delta first and takes it immediately if it is under 25% of
the raw size. Otherwise it also builds the full encoding and keeps the smaller one.
Delta chains are capped at depth 8 (`MaxChain`), so a read never decodes more than
8 objects for one segment.

Since the address depends only on content, not on encoding or history,
**identical segments are stored once**: across snapshots, across branches, and even
if two branches independently reach the same state.

**Tree** = list of `(segment index, hash, row count)` for 256 consecutive segments.
**Snapshot** = parents, message, time, mode, codebook hash, tree lists for vectors
and graph, graph header hash. Its hash is the **Merkle root**. Change one vector and
exactly one segment, one tree and the snapshot get new hashes. Everything else is
shared.

**Branch** = Pebble key `r/branch/<name>` → snapshot hash. Creating a branch is
one atomic key write: O(1), no data copied (zero-copy branching).

## 3. Commit paths

* `Commit(dataset, index)`: **full scan**. It hashes every segment and compares with
  the parent's hash. Equal segments are reused, changed ones are delta-encoded
  against the parent's version. Cost is O(N) hashing + O(changed) encoding. It also
  handles row reorder or compaction for free, because segments are keyed by ID, not
  by file offset.
* `CommitChanges(upserts, deletes, index)`: **incremental**. Only segments containing
  a changed ID, and graph segments containing an HNSW-dirty node, are loaded and
  re-encoded. The tests check that it produces **the same Merkle root as a full scan**.

Write protocol (`cas.Txn.Commit`):
1. append all new objects to the current pack file and fsync;
2. one Pebble batch (WAL, `Sync`) publishes every index entry and the branch ref.

A crash before step 2 leaves only unreferenced bytes in the pack. The repository
stays at the previous commit, and a test covers this.

## 4. Lossless vs lossy

**Lossless**: bit-exact. Unchanged vectors cost 0 bytes. Re-embedded vectors with
last-bit noise cost only their XOR delta, where the sign, exponent and high mantissa
byte-planes are almost all zero. Brand-new vectors cost about zstd of the shuffled floats.

**Lossy (semantic)**: `v ≈ c[coarse] + PQ(v − c[coarse])`. The code is 2 bytes for the
centroid id plus M bytes. The codebook is a CAS object shared by every snapshot that
uses it. Each row also keeps a 64-bit fingerprint of the *original* vector, so the
next commit can tell "unchanged" without keeping the original. Unchanged rows reuse
their code. **Drift detection** retrains the codebook if new data reconstructs more
than 1.25× worse than the training data (e.g. after a model upgrade).

## 5. HNSW (`internal/hnsw`)

* Algorithms 1, 2 and 4 of Malkov & Yashunin. Layer-0 capacity is 2M, upper layers M,
  and neighbour selection uses the diversity heuristic.
* **Deterministic levels**: `level = floor(−ln(U(hash(id, seed))) · 1/ln M)`.
* **Sorted neighbour lists**, so search is a pure function of (vectors, adjacency),
  which makes restore exact. The tests check that restored indexes return identical results.
* **Real deletes** (batch consolidation): each node that pointed at a deleted node keeps
  all its surviving edges and fills only the freed slots from the deleted nodes'
  neighbours. This keeps recall *and* keeps edge churn (and therefore TGIS delta size) small.
* **Dirty set**: every node whose adjacency changed since the last commit, which the
  incremental commit path consumes.
* `MarshalFlat()` writes an hnswlib-style flat file (fixed-size level-0 blocks with
  links, vector and label, then the upper layers). The byte-level baselines snapshot this file.

Validated on SIFT (20k): Recall@10 = 0.98 at ef=32 and 1.00 at ef=128.

## 6. Known limitations / future work

* No garbage collection of unreachable objects after branch deletion (it would be a
  mark-and-sweep over the Merkle DAG plus pack compaction).
* Single writer per repository; there is no cross-process lock file yet.
* HNSW build is single-threaded (parallel insert would need per-node locks).
* No gRPC server: the CLI and Go API cover the demo. Its `.proto` would mirror
  `repo.Commit/CommitChanges/CreateBranch/Checkout`.
* No FUSE mount and no `O_DIRECT`: these are Linux-specific and out of scope on Windows.
* Random edits spread across many segments, so incremental commit cost grows with
  the number of *touched segments*. Segment size (256) is a tunable trade-off.
