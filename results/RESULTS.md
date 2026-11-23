# VecDedup benchmark results

Environment: windows/amd64, 16 CPUs, go1.26.3. Generated 2026-11-21T13:25:34+05:30.

## 1. Vector snapshots: dedup ratio (base + 3 churned snapshots)

Cumulative ratio = all snapshot bytes / physical bytes. *Incr.* = ratio for snapshots 2–4 only (what each new snapshot costs).

### synthetic-50000x384

| scenario | full-copy | zstd | fixed128K+zstd | fastcdc8K+zstd | vecdedup-lossless | vecdedup-lossy(M=D/8) | lossy R@10 |
|---|---|---|---|---|---|---|---|
| edit-1% | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.47× (incr. 1.7×) | 3.04× (incr. 9.7×) | 4.47× (incr. 94.8×) | 80.03× (incr. 1063.4×) | 0.679 |
| edit-5% | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.01× (incr. 1.0×) | 1.79× (incr. 2.4×) | 3.94× (incr. 19.8×) | 73.14× (incr. 398.3×) | 0.657 |
| edit-10% | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.35× (incr. 1.5×) | 3.44× (incr. 10.0×) | 66.21× (incr. 226.3×) | 0.664 |
| edit-25% | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.06× (incr. 1.1×) | 2.48× (incr. 4.0×) | 51.53× (incr. 98.5×) | 0.643 |
| mixed-5% | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 2.04× (incr. 3.1×) | 4.12× (incr. 26.3×) | 75.18× (incr. 470.5×) | 0.649 |
| reembed-noise | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 2.63× (incr. 4.6×) | 24.73× (incr. 26.2×) | 0.665 |
| compaction-shuffle | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 4.56× (incr. 1681.4×) | 80.67× (incr. 1741.5×) | 0.679 |
| new-model | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.00× (incr. 1.0×) | 1.16× (incr. 1.2×) | 21.21× (incr. 21.2×) | 0.644 |

### sift-100kx128

| scenario | full-copy | zstd | fixed128K+zstd | fastcdc8K+zstd | vecdedup-lossless | vecdedup-lossy(M=D/8) | lossy R@10 |
|---|---|---|---|---|---|---|---|
| edit-1% | 1.00× (incr. 1.0×) | 3.48× (incr. 3.4×) | 3.52× (incr. 3.6×) | 6.90× (incr. 11.5×) | 14.16× (incr. 86.2×) | 64.09× (incr. 504.6×) | 0.551 |
| edit-5% | 1.00× (incr. 1.0×) | 3.05× (incr. 2.9×) | 2.91× (incr. 2.8×) | 3.33× (incr. 3.4×) | 9.97× (incr. 19.5×) | 61.06× (incr. 331.8×) | 0.457 |
| edit-10% | 1.00× (incr. 1.0×) | 2.68× (incr. 2.5×) | 2.57× (incr. 2.4×) | 2.55× (incr. 2.4×) | 7.28× (incr. 9.9×) | 58.32× (incr. 247.6×) | 0.447 |
| edit-25% | 1.00× (incr. 1.0×) | 2.04× (incr. 1.8×) | 1.99× (incr. 1.7×) | 1.89× (incr. 1.7×) | 4.03× (incr. 4.0×) | 36.27× (incr. 55.7×) | 0.192 |
| mixed-5% | 1.00× (incr. 1.0×) | 3.14× (incr. 3.0×) | 2.99× (incr. 2.9×) | 3.86× (incr. 4.2×) | 10.86× (incr. 24.6×) | 61.99× (incr. 355.1×) | 0.530 |
| reembed-noise | 1.00× (incr. 1.0×) | 2.23× (incr. 2.0×) | 2.17× (incr. 1.9×) | 1.99× (incr. 1.8×) | 3.43× (incr. 3.3×) | 18.87× (incr. 19.3×) | 0.572 |
| compaction-shuffle | 1.00× (incr. 1.0×) | 3.59× (incr. 3.6×) | 3.40× (incr. 3.4×) | 3.10× (incr. 3.1×) | 15.59× (incr. 576.7×) | 64.20× (incr. 598.4×) | 0.576 |
| new-model | 1.00× (incr. 1.0×) | 1.22× (incr. 1.0×) | 1.21× (incr. 1.0×) | 1.20× (incr. 1.0×) | 1.25× (incr. 1.0×) | 17.57× (incr. 17.5×) | 0.547 |

## 2. HNSW index snapshots (TGIS)

Bytes added by the snapshot taken *after* an incremental index update. Baselines snapshot the hnswlib-style flat index file (vectors + links).

| dataset | churn | touched nodes | ΔE+ | ΔE− | full copy | zstd | fixed128K+zstd | FastCDC+zstd | **VecDedup** | recall before / after / restored |
|---|---|---|---|---|---|---|---|---|---|---|
| sift-50000x128 | 1% | 9542 / 50051 (19%) | 15837 | 12146 | 31.5 MiB | 9.7 MiB | 10.0 MiB | 10.3 MiB | **301.5 KiB** | 0.990 / 0.995 / 0.995 |
| sift-50000x128 | 5% | 29833 / 50264 (59%) | 77316 | 58858 | 31.7 MiB | 10.1 MiB | 10.4 MiB | 11.3 MiB | **1020.2 KiB** | 0.990 / 0.993 / 0.993 |
| sift-50000x128 | 10% | 40855 / 50554 (81%) | 149213 | 112165 | 31.8 MiB | 10.5 MiB | 10.9 MiB | 11.8 MiB | **1.8 MiB** | 0.990 / 0.992 / 0.992 |

## 3. Lossy (semantic) mode: compression vs recall

Exact search over reconstructed vectors vs. ground truth on originals. Single snapshot; includes codebook.

| dataset | PQ M | bytes/vector | compression | Recall@10 | Recall@100 |
|---|---|---|---|---|---|
| synthetic-50000x384 | 24 | 26 | 31.7× | 0.603 | 0.730 |
| synthetic-50000x384 | 48 | 50 | 21.2× | 0.742 | 0.831 |
| synthetic-50000x384 | 96 | 98 | 12.7× | 0.885 | 0.935 |
| synthetic-50000x384 | 192 | 194 | 7.1× | 0.964 | 0.984 |
| sift-100kx128 | 8 | 10 | 24.4× | 0.432 | 0.549 |
| sift-100kx128 | 16 | 18 | 17.7× | 0.632 | 0.709 |
| sift-100kx128 | 32 | 34 | 11.5× | 0.766 | 0.832 |
| sift-100kx128 | 64 | 66 | 6.6× | 0.910 | 0.942 |

## 4. Latency

| vectors | dim | changed | initial commit | full-scan commit | incremental commit | branch | checkout |
|---|---|---|---|---|---|---|---|
| 100000 | 128 | 1000 | 311 ms | 133.4 ms | **96.2 ms** | **2.08 ms** | 45 ms |
| 1000000 | 128 | 1000 | 4030 ms | 1365.6 ms | **326.5 ms** | **0.81 ms** | 898 ms |
