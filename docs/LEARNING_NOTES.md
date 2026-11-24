# VecDedup: Learning Notes

Beginner-friendly, technical explanations of everything in the project, in the order you
need it. Every number in this file was measured on our own code (benchmarks in
`results/RESULTS.md`, or small demo runs that are described where they appear).

**How to use this file.** Read Part 1 for the idea, Part 2 for the background you need,
Part 3 for the core mechanism, Part 4 for each feature one at a time, and Part 5 for the
questions that tend to confuse people. `docs/STUDY_GUIDE.md` is the compact reference
version; `docs/PPT_CONTENT.md` has the slide content.

---

# Part 1: The big picture

## 1.1 One sentence

**VecDedup is a Git-like version control system for vector databases.** It saves every
version of the embeddings *and* their HNSW search index, but physically stores only what
changed between versions.

## 1.2 Version control or backup?

It is a **version control system**. It uses **deduplication** (the main trick of backup
systems) to keep the history cheap.

| | Backup system | Version control (what we built) |
|---|---|---|
| Purpose | recover after a disaster | keep a meaningful history and work with it |
| History | copies by date | commits with messages and parents |
| Branches | no | **yes**: instant experimental copies |
| Compare versions | usually no | **yes**: `diff` shows changed vectors and graph edges |
| Go back to a version | restore everything | **yes**: `checkout` any commit |
| Integrity check | sometimes | **yes**: `verify` re-checks every hash |
| Examples | restic, Borg, ZFS snapshots | Git, DVC, lakeFS |

Backup tools are what we *compare against*: they use chunk-based dedup (FastCDC), which
is our baseline. You can use VecDedup for backup (roll back to yesterday), but it is a
version control system.

## 1.3 The problem, step by step

1. An AI app (a RAG chatbot over documents) turns every document into a **vector**
   (embedding), a list of numbers such as 384 floats.
2. A **vector database** stores N such vectors plus an **HNSW index** for fast search.
   10M documents at 768-d is about 30 GB of vectors plus several GB of index.
3. The data changes daily: documents are edited, added, deleted, or the whole corpus is
   re-embedded.
4. Teams need history: **roll back** a bad update, **reproduce** old results (audit),
   **branch** to try a new model.
5. A full copy per snapshot is wasteful: 30 daily snapshots of 30 GB is 900 GB of mostly
   identical data.
6. The standard fix, chunk-based dedup, **performs badly on vector data** (Section 1.5).

## 1.4 Churn

**Churn = the fraction of vectors that changed between two consecutive snapshots.**

```
churn p = (#added + #modified + #deleted) / N
```

With N = 100,000: 1% churn = 1,000 vectors changed, 99,000 bit-identical to before.

We test each type of change separately, because each breaks byte-level tools differently:

| Scenario | What happens | Real-world cause |
|---|---|---|
| edit-p% | p% of rows get new vectors; IDs and order stay | documents edited and re-embedded |
| append | new rows with new IDs at the end | new documents |
| delete | rows removed; later rows shift position | documents removed |
| mixed-5% | 50% edit, 30% append, 20% delete of the 5% | a typical day |
| reembed-noise | **every** vector gets ~1e-6 relative noise | same model re-run on other hardware |
| compaction-shuffle | rows reordered (the benchmark also deletes 1%) | database compaction |
| new-model | every vector replaced by unrelated values | embedding-model upgrade |

## 1.5 Why chunk-based dedup fails on vector data

**How chunk dedup works.** Treat the file as bytes. Cut it into chunks (about 8 KB). Hash
each chunk (SHA-256). If the hash is already stored, store a reference instead.
**FastCDC** picks cut points from the content (a rolling hash), so boundaries survive
insertions.

**Root cause of every failure: it sees opaque bytes.** It doesn't know the bytes are rows
with IDs, floats that can be "almost equal", or graph edges.

**Failure 1: one changed row spoils a whole chunk.** An 8 KB chunk holds
8192 / 1536 ≈ 5 rows of 384-d. With churn p and r rows per chunk:

```
P(chunk unchanged) = (1 − p)^r        (r = 5)
 p = 1%  → 95% unchanged, 5% re-stored
 p = 10% → 59% unchanged, 41% re-stored
 p = 25% → 24% unchanged, 76% re-stored     (only 25% of rows changed!)
```
Measured incremental ratio (50k × 384): FastCDC 9.7× at 1%, 1.5× at 10%, 1.1× at 25%.

**Failure 2: re-embedding changes every float slightly.** Floating-point addition isn't
associative, `(a+b)+c ≠ a+(b+c)` in the last bit. GPUs sum in different orders depending on
hardware and batch size. The same text embedded twice can differ in the last bits:

```
0.12345678 = 3DFCD6E9
0.12345679 = 3DFCD6EA     (only the last 2 bits differ)
```
Meaning is unchanged, but byte comparison sees every chunk as new: ratio **1.0×**.

**Failure 3: reordering.** After compaction the same rows are in a different order, so
every chunk holds a different mix: **1.0×**, though no vector changed.

**Failure 4: the HNSW file.** One insert changes about M + 1 neighbour lists scattered over
the file (Lesson 3). 1% new nodes touched 19% of nodes on SIFT, so nearly every chunk has a
change: FastCDC added **10.3 MiB** for what is really 0.3 MiB of information.

Other tools: Git, DVC and lakeFS version whole files; ZFS dedups fixed blocks; Milvus and
Qdrant backups copy segment files; LanceDB shares unchanged row groups but has no
per-vector deltas and no index edge deltas.

---

# Part 2: Foundations (lessons)

## Lesson 1: Bits, bytes and float32

- A **bit** is 0 or 1. A **byte** is 8 bits (0–255). **Hex** writes a byte as two digits:
  `E9` = `1110 1001` = 233.
- A **float32** is a decimal number in **4 bytes (32 bits)**:

```
 1 bit    8 bits      23 bits
 sign     exponent    mantissa
value = (−1)^sign × 1.mantissa × 2^(exponent − 127)
```

**Worked example, 0.15625** = 1.25 × 2⁻³: sign 0; exponent 127 − 3 = 124 = `01111100`;
mantissa `.25` = `0100…0`. All bits: `0 01111100 0100…0` = hex **`3E200000`**.

**Nearly equal numbers share almost all bits** (the sign, exponent and top mantissa bits match):

```
0.12345678 = 3DFCD6E9 = 0011 1101 1111 1100 1101 0110 1110 1001
0.12345679 = 3DFCD6EA = 0011 1101 1111 1100 1101 0110 1110 1010
                                                            ^^ only these differ
```
Memory layout: the **least significant byte comes first**: `3DFCD6E9` is stored as
`E9 D6 FC 3D`.

## Lesson 2: Vectors, distance, nearest-neighbour search

A **vector** is a list of numbers (384 floats = 1,536 bytes). Similar texts give similar vectors.

**Squared Euclidean (L2) distance:**
```
a = [1,2,3], b = [2,2,5]   →   (1−2)² + (2−2)² + (3−5)² = 5      (smaller = more similar)
```
For unit-length vectors, `‖a−b‖² = 2 − 2·cos(a,b)`, so L2 ranking equals cosine ranking.

**Nearest-neighbour search:** find the k stored vectors closest to a query. **Brute force**
costs N × D (384 million multiply-adds for 1M × 384 per query), which is too slow.

## Lesson 3: HNSW (Hierarchical Navigable Small World)

**Step 1: a graph of neighbours.** Every vector is a node. Each node stores a list of the
IDs of a few nearby nodes (its **adjacency list**).

Teaching example (2-D points):
```
A(1,1) B(2,1) C(2,3) D(5,1) E(6,2) F(6,5) G(9,4) H(10,6)
A:[B,C]  B:[A,C,D]  C:[A,B]  D:[B,E]  E:[D,F]  F:[E,G,H]  G:[F,H]  H:[F,G]
```

**Step 2: search = walk toward the query.** Query q = (9,5). From the current node, move to the
neighbour closest to q; stop when none is closer. Start at B:

| Node | dist² to q | Neighbours' dist² | Move to |
|---|---|---|---|
| B | 65 | A 80, C 53, D 32 | D |
| D | 32 | B 65, E 18 | E |
| E | 18 | D 32, F 9 | F |
| F | 9 | E 18, G 1, H 2 | G |
| G | 1 | F 9, H 2 | stop |

Result G, the true nearest. (On 8 points this saves nothing; on a million points a walk takes
a few hundred distance computations.) A plain walk can **get stuck** (start at A, you stop at
C), so HNSW keeps a **beam** of the best `ef` candidates; `ef` trades accuracy for speed.

**Step 3: layers.** To avoid many small hops, HNSW adds sparse upper layers:
- Layer 0 has all nodes, up to **2M** neighbours each; upper layers have fewer nodes, up to **M**.
- A node's top layer is random: `P(level ≥ l) = M^(−l)` (M = 16: about 1 in 16 nodes reaches layer 1).
- Search: greedy descent through the top layers, then a beam search (width `ef`) on layer 0.
Example: with B and F on layer 1, search goes B → F directly, then G on layer 0.

**Step 4: insert.** New point X(8,5): (1) search for its nearest nodes G, F, H; (2) give X a
list of them; (3) **add X to each of their lists** (links go both ways); (4) if a list is over
capacity, drop the weakest edge (a heuristic prefers neighbours in different directions).
One insert changed 1 new list and 3 existing ones. With M = 16 it is about **17 lists**,
scattered through the file. This is why byte-chunking fails on indexes.

Delete is the reverse: lists that pointed to the deleted node must be repaired.

## Lesson 4: Hashes, content addressing and Merkle trees

**Hash (SHA-256):** any data → a fixed 32-byte fingerprint. Same input → same output; a tiny change →
a completely different output; two different inputs with the same hash is practically impossible.
`SHA-256("hello") = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824`.

**Content addressing:** store each object under the hash of its content. If the name exists, the data
is already stored (dedup). To read, re-hash and compare (detects corruption).

**Merkle tree:** hash the list of hashes. Four leaves a, b, c, d:
```
              root r = H(p1, p2)
             /                  \
     p1 = H(a, b)          p2 = H(c, d)
      /      \              /      \
     a        b            c        d
```
If b becomes b′: only b′, p1′ and the root are new; a, c, d, p2 are shared. One hash names the whole
state; equal hashes mean equal contents; **diff** compares from the root and skips equal subtrees.
Git uses the same structure. In VecDedup: snapshot → trees (up to 256 segments each) → segments.

**Snapshot:** an object holding its parent's hash, message, time and tree hashes. Because the parent
hash is inside, history is a chain. A **branch** is just a name pointing at a snapshot hash.

## Lesson 5: XOR, byte shuffle and zstd

Used to store a changed vector cheaply: `zstd( shuffle( XOR(new vector, old vector) ) )`.

**XOR** (bitwise): 1 where two bits differ. Applied to the 32-bit patterns of old and new floats:

```
old (hex)   new (hex)   XOR (hex)
3DFCD6E9    3DFCD6EA    00000003
BE4CCCCD    BE4CCCCC    00000001
3F19999A    3F19999A    00000000
3C23D70A    3C23D70B    00000001        → almost all zeros
```
XOR is **reversible**: `old XOR delta = new` (E9 XOR 03 = EA). We use XOR rather than
subtraction because float subtraction rounds, and we need bit-exact results.

**Byte-plane shuffle:** regroup the bytes so all first bytes come first, then all second bytes,
and so on:

```
as stored:   03 00 00 00 | 01 00 00 00 | 00 00 00 00 | 01 00 00 00
shuffled:    03 01 00 01   00 00 00 00   00 00 00 00   00 00 00 00
```
The zeros become one long run. Fully reversible.

**zstd** (Zstandard) compresses by two ideas:
1. **Repeats become "copy from earlier"** (LZ77). `"the cat " × 3` is stored as the literals
   `the cat ` plus *(offset 8, length 16)*. A run of 1,000 zeros is stored as one `00` plus
   *(offset 1, length 999)*.
2. **Common byte values get shorter codes** (entropy coding: Huffman for literals, FSE for the match
   instructions). Example: values with frequencies 50% / 25% / 25% get codes `0`, `10`, `11`, an average
   of 1.5 bits instead of 8.

Measured with our encoder:

| Input | Size | Compressed |
|---|---|---|
| 1,000 zero bytes | 1,000 | **16** |
| `"the cat "` × 125 | 1,000 | **31** |
| 1,000 random bytes (values 0–3) | 1,000 | 352 (ideal ≈ 250) |
| 1,000 random bytes (0–255) | 1,000 | **1,015** (grows) |

zstd cannot beat randomness, which is why brand-new float vectors only compress about 1.16×.
**zstd isn't the clever part. Our steps (delta, XOR, shuffle) make the data redundant, and zstd
squeezes it.** A match costs a few bytes, so runs shorter than about 4 bytes aren't worth matching;
that is why the shuffle (making zeros contiguous) helps zstd. Tiny inputs can even grow because zstd
adds a header (32 bytes of XOR data became 45).

Why zstd: fast to compress and especially to decompress, good ratio, RFC 8878, and a pure-Go library
(`klauspost/compress`). We did not benchmark other compressors.

---

# Part 3: The core mechanism

## 3.1 Stable IDs

Every vector comes with an **ID** from the application (its primary key: Qdrant point ID, Milvus
primary key, pgvector row ID). The ID identifies the **record**, not its content:

| | What it is | Changes when a document is edited? |
|---|---|---|
| ID (e.g. 5) | which record | **no** |
| vector | embedding of the current text | **yes** |

Editing is an **upsert**: ID 5 now has a new vector. If IDs were content hashes, an edit would look
like "delete one record, add another" in a random segment, ruining dedup.

**Assumption:** IDs are stable. If a pipeline re-numbers records by position on every rebuild, then
deleting one record changes the ID of every later record and the change cascades through every later
segment. That is an application-side issue (use a stable key such as document ID + chunk number).

## 3.2 Segments

```
segment number = id >> 8        (same as id ÷ 256, rounded down)
IDs 0–255 → segment 0, IDs 256–511 → segment 1, ...
```
The segment is a fixed **ID range**, not a fixed **row count**: it holds 0 to 256 rows. Membership
depends only on the ID itself, so adding, deleting or reordering rows never moves a vector into another
segment (no cascade).

## 3.3 Canonical form and hash

The segment is written in one fixed form (from `VecSeg.Hash`):

```
'V' | dim | number of rows | all IDs (sorted) | all float bits (in ID order)     → SHA-256
```
This is the segment's **address**. It depends only on content: not on file order, not on how it was
stored (full or delta), not on history. Toy example (dim 2, IDs 4 and 5):

```
56 | 02 00 00 00 | 02 00 00 00 | ID 4 (8 B) | ID 5 (8 B) | float bits ...   → SHA-256 → a1f3…
```
Edit row 5 and the whole hash changes.

## 3.4 What is stored in a snapshot (and where the saving is)

**A snapshot does not contain vectors. It is a recipe: a list of hashes.** The data lives in an object
store where each distinct object is stored **once**.

| Layer | Contents | Size |
|---|---|---|
| Snapshot (recipe) | "segment 0 is a1, segment 1 is b1, ..." | tiny (~40 B per segment) |
| Object store | the actual segment objects, each stored once | large |

Worked example (5 segments of 100 KB each; sizes illustrative):

```
Day 1: store a1 b1 c1 d1 e1 (500 KB).     C1 = [a1, b1, c1, d1, e1]
Day 2: seg 1 edited → b2 stored as a delta vs b1 (2 KB); new seg 5 → f1 (40 KB).
                                          C2 = [a1, b2, c1, d1, e1, f1]
Day 3: seg 1 → b3 (delta, 2 KB); seg 3 → d2 (delta, 4 KB).
                                          C3 = [a1, b3, c1, d2, e1, f1]
```

| Snapshot | Logical (full copy) | Physical added |
|---|---|---|
| C1 | 500 KB | 500 KB |
| C2 | 540 KB | 42 KB |
| C3 | 540 KB | 6 KB |
| Total | 1,580 KB | 548 KB (≈ 2.9×) |

Continuing for 30 snapshots: logical about 16,200 KB, physical about 714 KB, about 23×. The ratio improves
the more snapshots you keep.

**Cost of reading.** Day 3's segment 1 is a delta on b2, which is a delta on b1 (full). Reading follows
that chain and applies the changes. The chain is capped at **8** (`MaxChain`); after that a full copy is
written, bounding read cost. Base objects must be kept while anything depends on them.

Real numbers from the demo run (5,000 × 32-d, with HNSW): commit 1 added 755.02 KiB; commit 2 (5% mixed
churn, vectors and index) added 54.80 KiB; logical 2.07 MiB vs physical 815 KiB (2.6×). A snapshot over
20,000 vectors where all 79 segments were reused added **169 bytes**.

## 3.5 The delta, with every field filled in (toy segment)

Parent segment (dim 4, IDs 0–5) and new segment (IDs 0, 2, 3, 4, 5, 6, 7):

```
Parent                                       New
ID 0: [0.5,-0.25,0.75,0.125]                 ID 0 same
ID 1: [0.1,0.2,0.3,0.4]                      (ID 1 removed)
ID 2: [-0.6,0,0.9,0.3]                       ID 2 same
ID 3: [0.12345678,-0.2,0.6,0.01]             ID 3: [0.12345679,-0.19999999,0.6,0.010000001]  (noise)
ID 4: [0.3,0.3,-0.5,0.7]                     ID 4: [-0.8,0.45,0.2,0.05]                      (real edit)
ID 5: [0.9,0.1,0,-0.4]                       ID 5 same
                                             ID 6: [0.33,0.66,-0.99,0.11]                    (added)
                                             ID 7: [-0.1,0.8,0.4,-0.7]                       (added)
```

**Classify each ID by comparing parent and new, matching by ID:**

| ID | Class | What is stored |
|---|---|---|
| 0, 2, 5 | unchanged | nothing |
| 1 | removed | only its ID |
| 3, 4 | changed | XOR of new bits against old bits |
| 6, 7 | added | the raw floats |

**XOR for the changed rows:**
```
ID 3:  00000003 00000001 00000000 00000001        (noise → almost zeros)
ID 4:  81D55557 007FFFFC 814CCCCD 027FFFFE        (real edit → random-looking)
```
The encoder does not know which is which and doesn't need to: XOR is never worse than storing the new
bits and is far better when the vector barely moved.

**Shuffle** of the 32 XOR bytes: byte-0 plane `03 01 00 01 | 57 FC CD FE`, byte-1 plane
`00 00 00 00 | 55 FF CC FF`, byte-2 plane `00 00 00 00 | D5 7F 4C 7F`, byte-3 plane `00 00 00 00 | 81 00 81 02`.

**The final object (134 bytes, encoded then decoded and verified bit-identical):**

| Field | Value | Bytes |
|---|---|---|
| base hash | parent segment's hash | 32 |
| depth | 1 | 1 |
| dim | 4 | 1 |
| removed | `01 01` (count 1, ID 1) | 2 |
| changed | `02 03 01` (count 2, ID 3, then +1) | 3 |
| added | `02 06 01` (count 2, ID 6, then +1) | 3 |
| XOR blob | zstd(shuffle(XOR data)) | 1 + 45 |
| added blob | zstd(shuffle(raw rows)) | 1 + 45 |
| **Total** | | **134** |

IDs are stored as **differences between sorted values** (varint: under 128 takes 1 byte, under 16,384
takes 2). On the toy, zstd makes 32 bytes into 45 because of its fixed header. At real size it pays off.

**Encoder rule** (`EncodeVec`): build the delta; if it is under **25%** of the raw size use it. Otherwise
also build the full encoding and keep the smaller. A delta can sit on a delta, capped at depth 8.

## 3.6 Where the space is saved (real size: 256 rows × 384 floats = 393,216 bytes)

| Technique | Measured |
|---|---|
| Unchanged rows aren't stored | 3 edited rows: delta **4,675 B** vs **340,505 B** in full (73×) |
| XOR against the old version | the same 3 rows with noise only: **1,201 B** (about 4× less than edited) |
| Byte shuffle before zstd | all 256 rows re-embedded: XOR+zstd 118,394 B vs XOR+shuffle+zstd **86,288 B** (4.6× vs raw) |
| zstd on brand-new data | zstd alone 393,238 B (no gain); shuffle+zstd 340,181 B (**1.16×**) |
| Delta-coded IDs | 1 byte per ID instead of 8 |
| mixed (1 removed, 2 changed, 2 added) | **6,220 B** vs 342,425 B in full |

**Lossless ceiling:** the low mantissa bits of embeddings are close to random, so only sign and exponent
bytes compress. Published results: about 1.2–1.5×. We measure 1.16–1.25×. Big ratios come from
redundancy **between snapshots**, not from compressing new floats.

---

# Part 4: The six features, one at a time

## Feature 1: Small edits cost almost nothing

**Goal:** a new snapshot should cost about as much as the changed rows.

**How:** stable IDs → segments (`id >> 8`) → canonical hash → compare with the parent's hash for that
segment: same = reuse (0 bytes), different = delta, new = full → tree → snapshot.

**Numbers** (50,000 × 384, 76.8 MB per full copy, 1% churn = 500 edits). There are about 196 segments. The
chance a segment has no edit is 0.99²⁵⁶ ≈ 7.6%, so about 181 segments get a delta of roughly 2.6 rows each.
Estimated cost: 500 × 1.3 KB ≈ 650 KB for the rows plus bookkeeping; measured about 0.8 MB per snapshot, so
**94.8×** (FastCDC 9.7×).

VecDedup stores about the changed rows, so the ratio is roughly 1 ÷ churn: 95× at 1%, 20× at 5%, 10× at 10%,
4× at 25%. FastCDC stores whole chunks, so it falls off much faster.

## Feature 2: Re-embedding noise (the last-bit problem)

**We did not fix the noise; we made it cheap to store while keeping every bit exact.** XOR against the old
version turns "almost equal" into "mostly zeros"; shuffle makes the zeros contiguous; zstd collapses them.

**Result:** **4.6×** (synthetic) and **3.3×** (SIFT) vs 1.0× for FastCDC. Whole-segment measurement: 86,288 B
vs 393,216 B raw.

**Limits:** modest, because the low bits really did change. The noise in our benchmark is **simulated**
(about 1e-6 relative), not measured on real different-GPU runs. We did **not** build a "treat vectors within a
tolerance as unchanged" option; that would be lossy.

## Feature 3: Reordering and compaction are free

A database rewrites its files and the rows end up in a different order; no vector changed. A chunk tool sees
all-new chunks. VecDedup groups by ID and sorts rows by ID **before hashing**, so file order is discarded.

**Measured** (20,000 × 384, 79 segments):

| | Full | Delta | Reused | New objects | New bytes |
|---|---|---|---|---|---|
| Commit 1 | 79 | 0 | 0 | 81 | 26.6 MB |
| Commit 2 (shuffled) | 0 | 0 | **79** | **1** | **169 B** |

FastCDC ratio after the shuffled file: **1.00** (it reused nothing). The benchmark's **1681×** is a scenario
that deletes 1% of the rows and then shuffles: the shuffle is free, and the deletes cost small removed-ID lists
(about 46 KB per snapshot; consistent with about 181 tiny deltas). **Limit:** needs stable IDs; if the
compaction re-numbers IDs, nothing is reused. Restored rows come back sorted by ID, not in the original order.

## Feature 4: The HNSW index as edge changes

**What's stored:** per node, its level and its per-layer neighbour lists (sorted); nodes are grouped into
256-ID segments with the same hash/delta scheme.

**What the delta stores:** removed node IDs; added nodes (full record); modified node IDs; for each modified
node and layer, **ΔE⁺ (edges added)** and **ΔE⁻ (edges removed)**.

**It does not use XOR.** XOR compares position by position, and a neighbour list has no fixed positions:

```
old [3,9,12], new [3,12] (padded [3,12,0]) → XOR = [0,5,12]   (one removal gave two garbage numbers)
set difference → ΔE⁻ = [9], ΔE⁺ = []                           (exactly what changed)
```

| Data | Difference operation |
|---|---|
| float vector that moved slightly | XOR of bit patterns |
| set of rows in a segment | removed / changed / added IDs |
| node's neighbour list | ΔE⁺ and ΔE⁻ (set difference) |

Example: `[12,90,301,4410,7002] → [12,90,301,5120,7002]` gives ΔE⁺ = [5120], ΔE⁻ = [4410], about 6 bytes.
IDs are stored as differences (`[12,90,301,4410]` → 12, +78, +211, +4109 → 6 bytes vs 32 as 8-byte numbers).
The saving here comes from **listing only the differences**; zstd adds little because neighbour IDs are fairly random.

**Real output of our code (8 points, M = 2, insert X at (8,5)):** the index reported 4 dirty nodes
(E, F, G, X). E layer 1: `[B] → [B,X]`, F: `[E,G] → [E,G,X]`, G: `[F,H] → [F,H,X]`, X new with `[F,G]`.
DiffGraph: 1 node added, 3 modified, 6 edges added, 0 removed. Toy delta 71 B vs 81 B full.

**Bigger run** (10,000 nodes, 32-d, M = 16, insert 100 = 1%): **953 of 10,100 nodes touched (9%)**; all 40 segments
touched; ΔE⁺ 1,786 (includes the new nodes' own lists), ΔE⁻ 145; whole graph in full 254,256 B vs delta
**8,160 B (about 31×)**.

**Benchmark on SIFT 50k** (bytes added by one snapshot):

| Churn | Nodes touched | ΔE⁺ | ΔE⁻ | zstd | FastCDC+zstd | **VecDedup** | Recall@10 before / after / restored |
|---|---|---|---|---|---|---|---|
| 1% | 19% | 15,837 | 12,146 | 9.7 MiB | 10.3 MiB | **301.5 KiB** | 0.990 / 0.995 / 0.995 |
| 5% | 59% | 77,316 | 58,858 | 10.1 MiB | 11.3 MiB | **1,020 KiB** | 0.990 / 0.993 / 0.993 |
| 10% | 81% | 149,213 | 112,165 | 10.5 MiB | 11.8 MiB | **1.8 MiB** | 0.990 / 0.992 / 0.992 |

FastCDC (10.3 MiB) added more than plain zstd of the whole file (9.7 MiB): it saved nothing. At 1%, VecDedup is
about 35× smaller than FastCDC.

**Why chunking fails here:** a 128-d node block is about 650 bytes, so an 8 KB chunk holds about 12 nodes
(estimate). With 19% of nodes touched, a chunk is untouched with probability about 0.81¹² ≈ 8%.

**Design choices that make the delta small and restore exact:**
1. neighbour lists are **sorted**, giving one canonical form;
2. node levels come from **`hash(id, seed)`**, so a node always lands on the same level;
3. the index tracks **dirty nodes**, so commits only re-encode touched segments;
4. **delete repair keeps surviving edges** and fills only the freed slots (my first version re-pruned whole
   lists: −77k/+17k edges, hurt recall and made deltas large);
5. a checkout **restores the index from stored lists without rebuilding**; tests confirm identical search results.

## Feature 5: Lossy mode (product quantization)

**Goal:** shrink a 384-d vector from 1,536 bytes to about 50 bytes, accepting an approximation. In lossy mode
checkout returns **reconstructed** vectors (bit-exact history no longer holds).

**k-means** builds the table of typical vectors: pick K centroids, assign each point to its nearest, move each
centroid to the average of its points, repeat. Toy (K = 2): a(1,1), b(1,2), c(2,1), d(8,8), e(9,8), f(8,9) →
centroids (1.33,1.33) and (8.33,8.33). Storing (1.2,1.5) as centroid 0 decodes to (1.33,1.33), error 0.046.

**Product quantization:** split the vector into M pieces, each with its own table of 256 entries and 1 byte per
piece; M pieces give 256^M combinations from 256 × M entries. Toy (4 numbers, 2 pieces, 4 entries each):
v = (0.85,0.15 | 0.45,0.55) → code [1,1] → decoded (0.9,0.1,0.5,0.5), error 0.01. 8 table entries represent
4 × 4 = 16 vectors.

**Residual (IVFADC):** first take the nearest **coarse centroid** (K = 256, 2 bytes), then PQ-encode the
**residual** `v − centroid`, which is small. Toy: v = (0.87,0.13,0.43,0.57), coarse C0 = (0.8,0.2,0.4,0.6),
residual (0.07,−0.07,0.03,−0.03), code [0,1,0], decoded (0.9,0.1,0.4,0.6), error **0.0036** vs **0.0116** for plain PQ.

**Code = 2 + M bytes.** Default K = 256, M = D/8 (384-d: M = 48, 50 bytes).

**Size arithmetic** (50,000 × 384, M = 48): codes + 8-byte fingerprint = 50,000 × 58 = 2.9 MB; codebook
≈ 2 × 256 × 384 × 4 = 0.79 MB; total 3.69 MB vs 76.8 MB → **20.8×** (measured 21.2×). Same arithmetic gives
30.9× / 12.6× / 7.1× for M = 24 / 96 / 192 (measured 31.7× / 12.7× / 7.1×). A lossy re-embed costs
≈ 58 B per vector → 26× (measured 26.2×).

**In the versioned store:**
1. the codebook is one stored object shared by every snapshot using it;
2. segments hold `(id, fingerprint, code)`;
3. **fingerprint** = 64-bit xxhash of the original vector's bits; since originals are discarded, the next commit
   hashes each incoming vector and compares: equal → reuse the old code; different → re-encode;
4. deltas hold removed IDs and upserted rows.

**Drift detection:** on each lossy commit, sample up to 2,000 vectors and compute
`relMSE = Σ‖v − v̂‖² / Σ‖v‖²` and `relCoarse = Σ‖v − c*‖² / Σ‖v‖²`. If either exceeds **1.25×** its training
value (`DriftFactor`), retrain the codebook. (A test rescaling data by 0.01 once dropped recall to about 0.01
with the stale codebook; this fixed it.)

**Quality = Recall@k** = |returned ∩ true top-k| / k. Example (k = 5): true {a,b,c,d,e}, returned {a,c,x,d,y} →
3/5 = 0.6. Measured by exact search on reconstructed vectors vs exact search on originals:

| Data | M | Compression | Recall@10 |
|---|---|---|---|
| synthetic 384-d | 24 | 31.7× | 0.603 |
| synthetic 384-d | 48 | 21.2× | 0.742 |
| synthetic 384-d | 96 | 12.7× | 0.885 |
| synthetic 384-d | 192 | 7.1× | 0.964 |
| SIFT 128-d | 8 | 24.4× | 0.432 |
| SIFT 128-d | 16 | 17.7× | 0.632 |
| SIFT 128-d | 32 | 11.5× | 0.766 |
| SIFT 128-d | 64 | 6.6× | 0.910 |

**Limits:** recall is **not bounded**; the spec's "≤0.5% loss" is **not met**. No re-ranking with originals
(future work). Recall wasn't measured through an HNSW on top.

## Feature 6: Branches, diff, verify, checkout and crash safety

**Snapshot object:** parents, time, message, dim, count, mode, codebook, vector trees, graph header, graph trees.
Its hash is the Merkle root; `log` follows `parents`.

**Branch:** one entry `branch/<name> → snapshot hash` written in a single atomic transaction. It checks that the
name is free and the snapshot exists. Nothing is copied. Measured: 5.1 ms in the CLI demo, 0.8–2 ms in the
in-process benchmark. A commit on a branch moves only that branch's pointer. `main~1` means one step back via
`parents`. **Limits:** no merge; deleting a branch doesn't free objects (no GC).

**Diff:** compare the two snapshots' trees; skip trees with equal hashes; compare segment lists; skip segments
with equal hashes; load only differing segments and compare rows by ID. Demo output:

```
d460a2f063cd -> 119f2724e71b
  vectors: +74 added  -49 removed  ~125 changed
  graph:   nodes +74 -49 ~3310   edges ΔE+ 8187  ΔE- 7319
  merkle:  2 trees compared, 0 skipped by hash       (tiny demo: one tree each)
```
Skipping depends on locality: with random edits nearly every tree changes, but the segment level skips about 97%
(1,000 of 39,000 segments differ at 1,000 random edits on 10M).

**Verify:** for each snapshot, (1) hash each tree and compare with its address; (2) **fully decode every segment**
(cache cleared), recompute the content hash and row count, compare with the tree; (3) check the snapshot hash.
Step 2 also proves that delta decoding reproduces the exact content. Every object read is also CRC32-checked.
**Tamper test:** flipping one bit in a copy of the pack file → `error: ... cas: checksum mismatch for e6996d87e0db`;
the original copy → `ok` for both snapshots. Corruption is detected, not repaired.

**Checkout:** read snapshot → trees → segments, follow delta chains (≤ 8), rebuild the dataset, and load the stored
adjacency back into the index (no rebuild). Demo: 5,000 vectors + index restored in 17 ms; the restored file was
byte-identical to the original input (which was already in ID order). Benchmark: 45 ms at 100k, 898 ms at 1M.

**Crash safety (write protocol):** (1) append objects to a pack and **fsync**; (2) one Pebble batch with Sync
records object locations and moves the branch pointer. A crash in between leaves only unreferenced bytes; the repo
stays at the previous commit. **Tested by simulation** (an abandoned transaction is invisible; corrupted bytes are
detected), not by real power-failure testing.

**Commit paths:** `Commit` scans and hashes every segment; `CommitChanges` touches only segments containing changed
IDs. On 1M vectors with 1,000 changes: **327 ms vs 1.37 s**; a test checks both give the **same Merkle root**.

---

# Part 5: Questions that tend to confuse people

**Why not one object per vector?** The snapshot would need a 32-byte hash plus ID per vector in every snapshot. For
1M vectors and 1% churn: about 40 MB of bookkeeping vs about 15 MB of changed data. For PQ codes (about 50 B) a
32-byte hash is 64% overhead; for a 128-byte neighbour list it is 25%. Millions of tiny objects also slow commits
and checkouts, and zstd works better on larger blocks. With 256-ID segments the recipe is about 156 KB for 1M vectors.
(This is arithmetic, not a measured per-vector experiment.)

**But then a changed segment is stored again, like FastCDC?** No. Grouping is for **addressing**; storage is a
**delta**. With 1% random churn about 92% of segments change, but each delta holds only about 2.6 rows. FastCDC stores
the whole chunk.

**What is generic and what is vector-specific?**
- Generic (not new; Git, Dolt, lakeFS, Iceberg do it): content addressing, Merkle trees, snapshots, branches, segmenting
  keyed rows, "same hash means store nothing".
- Vector-specific (our contribution): XOR on float32 bits plus 4-byte byte-plane shuffle; the re-embedding-noise design;
  HNSW stored as ΔE⁺/ΔE⁻ with dirty tracking, deterministic levels and delete repair; lossy PQ with shared codebook,
  fingerprints and drift detection.
- Chunking works on any data; ours needs a **key** (the ID), which vector databases always have.

**What is SIFT?** SIFT1M is a standard public benchmark: 1,000,000 vectors of 128 numbers, each a descriptor of an image
patch (Scale-Invariant Feature Transform), with 10,000 queries and true neighbours (TEXMEX collection, INRIA). We downloaded
the first 100,000 vectors (52 MB) and used it to validate our HNSW (Recall@10: 0.936 / 0.978 / 0.992 / 1.000 at ef = 16 /
32 / 64 / 128 on 20k vectors), for the vector benchmark (100k × 128), the index benchmark (50k) and the lossy curve.
It is real data, but image descriptors, not text embeddings; its values are small whole numbers stored as floats, so plain
compression does unusually well (about 3×).

**Is "not fixing the last-bit noise" a weakness?** It is a design choice: we keep bit-exactness, so history is exactly
reproducible. A tolerance-based mode could save more but would be lossy.

**Why Go, Pebble, zstd?** Single static binary and good concurrency; Pebble gives RocksDB-style LSM semantics with atomic
batches without cgo (RocksDB is hard to build on Windows); zstd is fast with a pure-Go library.

---

# Part 6: Honest limits and corrections to the original spec

| Original claim | Status | What we can defend |
|---|---|---|
| "FastCDC gets 0% dedup" | false in general | 9.7× at 1% edits, ~1× at 25%, 1.0× on re-embed noise, compaction and index updates |
| > 4× on incremental updates | met | lossless 4.0–94.8× (synthetic), 4.0–86× (SIFT) at 1–25% churn |
| lossless 8–16× | impossible for new data | 1.16–1.25× measured; ceiling about 1.2–1.5× |
| ≤ 0.5% recall loss, "bounded" | not met, not true | 7.1× → R@10 0.964; 32× → 0.603 |
| HNSW snapshots via edge deltas | met | 0.3 MiB vs 10.3 MiB at 1%; identical results after restore |
| < 200 ms commit on 10M vectors | not shown | 327 ms at 1M (incremental); 10M not measured |
| < 50 ms branch | met | 0.8–2 ms |
| RocksDB | changed | Pebble (pure Go, RocksDB-like) |
| gRPC server, FUSE mount | not built | future work |

Other limits: no merge, no garbage collection, single writer, single-threaded HNSW build, re-embedding noise simulated
(not measured on real GPUs), crash safety tested by simulation, synthetic data supplemented by SIFT (images) rather
than real text embeddings.

---

# Part 7: Self-check questions

1. Why does one changed vector in a chunk force the whole chunk to be stored again? Write the formula.
2. Why is the XOR of two nearly equal floats mostly zeros? Why can we recover the new value exactly from it?
3. Why do we shuffle bytes before zstd? What does zstd do with the result?
4. Why is the segment hash computed over sorted canonical content and not the file bytes?
5. A snapshot contains only hashes. Where does the saving come from?
6. Why can't XOR be used on a neighbour list? What do we use instead?
7. Why does 1% new nodes touch far more than 1% of neighbour lists?
8. What does a product-quantization code contain, and why does the residual help?
9. What does the fingerprint do in lossy mode?
10. What does `verify` check that a plain checksum doesn't?
11. Why does creating a branch take about 1 ms regardless of size?
12. Which claims of the original spec were not met, and what did we measure instead?
