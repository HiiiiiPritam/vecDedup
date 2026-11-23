// Package hnsw is a deterministic implementation of Hierarchical Navigable Small
// World graphs (Malkov & Yashunin, arXiv:1603.09320) designed to be snapshotted.
//
// Differences from hnswlib that matter for VecDedup:
//   - A node's level is derived from hash(id, seed), not from an RNG stream, so the
//     same id always lands on the same level regardless of insertion order.
//   - Neighbour lists are kept sorted (by external id). Search results are therefore
//     a pure function of (vectors, adjacency), which makes snapshot → restore exact.
//   - Deletes are real (FreshDiskANN-style batch consolidation with neighbour
//     repair), not just tombstones, and freed slots are reused (like hnswlib's
//     allow_replace_deleted) so the flat file layout stays stable.
//   - The index tracks which nodes' adjacency changed since the last snapshot
//     ("dirty" set), so commits only re-encode touched graph segments.
package hnsw

import (
	"container/heap"
	"encoding/binary"
	"errors"
	"math"
	"sort"

	"vecdedup/internal/vec"
)

// Config holds construction parameters.
type Config struct {
	M              int    // max neighbours on layers ≥ 1 (layer 0 allows 2M)
	EfConstruction int    // beam width during insertion
	Seed           uint64 // level-assignment seed
}

// DefaultConfig returns common parameters (M=16, efC=100).
func DefaultConfig() Config { return Config{M: 16, EfConstruction: 100, Seed: 42} }

const noSlot = math.MaxUint32

// Index is an HNSW graph over L2-squared distance (equivalent to cosine ranking
// for unit-norm vectors).
type Index struct {
	Cfg Config
	Dim int

	mL       float64
	vecs     []float32    // slot*Dim
	ids      []uint64     // slot -> external id
	levels   []int32      // slot -> top level, -1 = free slot
	links    [][][]uint32 // slot -> layer -> neighbour slots (sorted by external id)
	idmap    map[uint64]uint32
	free     []uint32
	entry    uint32
	maxLevel int

	dirty map[uint64]bool
}

// New creates an empty index.
func New(dim int, cfg Config) *Index {
	if cfg.M < 2 {
		cfg.M = 2
	}
	if cfg.EfConstruction < cfg.M {
		cfg.EfConstruction = cfg.M
	}
	return &Index{Cfg: cfg, Dim: dim, mL: 1 / math.Log(float64(cfg.M)), idmap: map[uint64]uint32{}, entry: noSlot, maxLevel: -1, dirty: map[uint64]bool{}}
}

// Len returns the number of live nodes.
func (h *Index) Len() int { return len(h.idmap) }

// Contains reports whether id is in the index.
func (h *Index) Contains(id uint64) bool { _, ok := h.idmap[id]; return ok }

func (h *Index) maxConn(layer int) int {
	if layer == 0 {
		return 2 * h.Cfg.M
	}
	return h.Cfg.M
}

func (h *Index) vecOf(s uint32) []float32 { return h.vecs[int(s)*h.Dim : (int(s)+1)*h.Dim] }

// Vector returns the stored vector for id.
func (h *Index) Vector(id uint64) ([]float32, bool) {
	s, ok := h.idmap[id]
	if !ok {
		return nil, false
	}
	return h.vecOf(s), true
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// LevelFor returns the deterministic level of an id: floor(-ln(U) * mL).
func (h *Index) LevelFor(id uint64) int {
	u := float64(splitmix64(id^h.Cfg.Seed)>>11+1) / float64(1<<53) // (0,1]
	l := int(-math.Log(u) * h.mL)
	if l > 30 {
		l = 30
	}
	return l
}

// ---------------------------------------------------------------------------
// priority queues
// ---------------------------------------------------------------------------

type cand struct {
	d float32
	s uint32
}

type minHeap []cand

func (q minHeap) Len() int            { return len(q) }
func (q minHeap) Less(i, j int) bool  { return q[i].d < q[j].d }
func (q minHeap) Swap(i, j int)       { q[i], q[j] = q[j], q[i] }
func (q *minHeap) Push(x interface{}) { *q = append(*q, x.(cand)) }
func (q *minHeap) Pop() interface{}   { o := *q; x := o[len(o)-1]; *q = o[:len(o)-1]; return x }

type maxHeap []cand

func (q maxHeap) Len() int            { return len(q) }
func (q maxHeap) Less(i, j int) bool  { return q[i].d > q[j].d }
func (q maxHeap) Swap(i, j int)       { q[i], q[j] = q[j], q[i] }
func (q *maxHeap) Push(x interface{}) { *q = append(*q, x.(cand)) }
func (q *maxHeap) Pop() interface{}   { o := *q; x := o[len(o)-1]; *q = o[:len(o)-1]; return x }

// searchLayer is Algorithm 2 of the paper: beam search with width ef. Returns
// candidates sorted ascending by distance (ties broken by external id).
func (h *Index) searchLayer(q []float32, eps []uint32, ef, layer int) []cand {
	visited := make(map[uint32]struct{}, ef*4)
	var C minHeap
	var W maxHeap
	for _, e := range eps {
		if _, ok := visited[e]; ok {
			continue
		}
		visited[e] = struct{}{}
		d := vec.L2Sq(q, h.vecOf(e))
		heap.Push(&C, cand{d, e})
		heap.Push(&W, cand{d, e})
	}
	for C.Len() > 0 {
		c := heap.Pop(&C).(cand)
		if W.Len() >= ef && c.d > W[0].d {
			break
		}
		if int(h.levels[c.s]) < layer {
			continue
		}
		for _, n := range h.links[c.s][layer] {
			if _, ok := visited[n]; ok {
				continue
			}
			visited[n] = struct{}{}
			d := vec.L2Sq(q, h.vecOf(n))
			if W.Len() < ef || d < W[0].d {
				heap.Push(&C, cand{d, n})
				heap.Push(&W, cand{d, n})
				if W.Len() > ef {
					heap.Pop(&W)
				}
			}
		}
	}
	out := make([]cand, len(W))
	copy(out, W)
	h.sortCands(out)
	return out
}

func (h *Index) sortCands(cs []cand) {
	sort.Slice(cs, func(i, j int) bool {
		if cs[i].d != cs[j].d {
			return cs[i].d < cs[j].d
		}
		return h.ids[cs[i].s] < h.ids[cs[j].s]
	})
}

// selectHeuristic is Algorithm 4: keep a candidate only if it is closer to the
// base point than to every neighbour already selected. cands must be sorted.
func (h *Index) selectHeuristic(cands []cand, m int) []uint32 {
	out := make([]uint32, 0, m)
	for _, c := range cands {
		if len(out) >= m {
			break
		}
		good := true
		cv := h.vecOf(c.s)
		for _, r := range out {
			if vec.L2Sq(cv, h.vecOf(r)) < c.d {
				good = false
				break
			}
		}
		if good {
			out = append(out, c.s)
		}
	}
	return out
}

func (h *Index) sortLinks(l []uint32) {
	sort.Slice(l, func(i, j int) bool { return h.ids[l[i]] < h.ids[l[j]] })
}

func (h *Index) markDirty(s uint32) { h.dirty[h.ids[s]] = true }

// ---------------------------------------------------------------------------
// Insert / Delete / Update
// ---------------------------------------------------------------------------

// Insert adds a new id (or replaces the vector of an existing one).
func (h *Index) Insert(id uint64, v []float32) {
	if len(v) != h.Dim {
		panic("hnsw: dim mismatch")
	}
	if _, ok := h.idmap[id]; ok {
		h.DeleteBatch([]uint64{id})
	}
	level := h.LevelFor(id)
	var s uint32
	if n := len(h.free); n > 0 {
		s = h.free[n-1]
		h.free = h.free[:n-1]
		copy(h.vecOf(s), v)
		h.ids[s] = id
		h.levels[s] = int32(level)
	} else {
		s = uint32(len(h.ids))
		h.vecs = append(h.vecs, v...)
		h.ids = append(h.ids, id)
		h.levels = append(h.levels, int32(level))
		h.links = append(h.links, nil)
	}
	h.links[s] = make([][]uint32, level+1)
	h.idmap[id] = s
	h.markDirty(s)

	if h.entry == noSlot {
		h.entry, h.maxLevel = s, level
		return
	}
	q := h.vecOf(s)
	ep := []uint32{h.entry}
	for l := h.maxLevel; l > level; l-- {
		ep = []uint32{h.searchLayer(q, ep, 1, l)[0].s}
	}
	for l := min(level, h.maxLevel); l >= 0; l-- {
		W := h.searchLayer(q, ep, h.Cfg.EfConstruction, l)
		nb := h.selectHeuristic(W, h.Cfg.M)
		h.links[s][l] = append([]uint32(nil), nb...)
		h.sortLinks(h.links[s][l])
		for _, n := range nb {
			h.addLink(n, s, l)
		}
		ep = ep[:0]
		for _, c := range W {
			ep = append(ep, c.s)
		}
	}
	if level > h.maxLevel {
		h.entry, h.maxLevel = s, level
	}
}

// addLink adds edge n -> s on layer l, pruning n's list with the heuristic if full.
func (h *Index) addLink(n, s uint32, l int) {
	h.markDirty(n)
	lst := append(h.links[n][l], s)
	if len(lst) > h.maxConn(l) {
		nv := h.vecOf(n)
		cs := make([]cand, len(lst))
		for i, x := range lst {
			cs[i] = cand{vec.L2Sq(nv, h.vecOf(x)), x}
		}
		h.sortCands(cs)
		lst = h.selectHeuristic(cs, h.maxConn(l))
	}
	h.sortLinks(lst)
	h.links[n][l] = lst
}

// DeleteBatch removes ids and repairs every node that pointed at them: its remaining
// neighbours plus the deleted nodes' neighbours become candidates, re-selected with
// the heuristic (FreshDiskANN-style consolidation). One pass over the graph.
func (h *Index) DeleteBatch(ids []uint64) {
	dead := map[uint32]bool{}
	for _, id := range ids {
		if s, ok := h.idmap[id]; ok {
			dead[s] = true
		}
	}
	if len(dead) == 0 {
		return
	}
	for s := range h.links {
		su := uint32(s)
		if h.levels[s] < 0 || dead[su] {
			continue
		}
		for l, lst := range h.links[s] {
			hit := false
			for _, n := range lst {
				if dead[n] {
					hit = true
					break
				}
			}
			if !hit {
				continue
			}
			// Keep every surviving edge untouched (minimal edge churn = small TGIS
			// deltas, and no loss of good existing links). Fill only the freed slots
			// with the dead nodes' neighbours, chosen by the diversity heuristic
			// relative to the edges we kept.
			keep := make([]uint32, 0, len(lst))
			inList := map[uint32]bool{su: true}
			for _, n := range lst {
				if !dead[n] {
					keep = append(keep, n)
					inList[n] = true
				}
			}
			free := len(lst) - len(keep)
			sv := h.vecOf(su)
			var cs []cand
			for _, n := range lst {
				if !dead[n] || l >= len(h.links[n]) {
					continue
				}
				for _, nn := range h.links[n][l] {
					if !dead[nn] && !inList[nn] && int(h.levels[nn]) >= l {
						inList[nn] = true
						cs = append(cs, cand{vec.L2Sq(sv, h.vecOf(nn)), nn})
					}
				}
			}
			h.sortCands(cs)
			added := 0
			for _, c := range cs {
				if added >= free {
					break
				}
				good := true
				cv := h.vecOf(c.s)
				for _, r := range keep {
					if vec.L2Sq(cv, h.vecOf(r)) < c.d {
						good = false
						break
					}
				}
				if good {
					keep = append(keep, c.s)
					added++
				}
			}
			for _, c := range cs { // heuristic too strict: top up with nearest
				if added >= free {
					break
				}
				if !containsSlot(keep, c.s) {
					keep = append(keep, c.s)
					added++
				}
			}
			h.sortLinks(keep)
			h.links[s][l] = keep
			h.markDirty(su)
		}
	}
	for s := range dead {
		h.dirty[h.ids[s]] = true
		delete(h.idmap, h.ids[s])
		h.levels[s] = -1
		h.links[s] = nil
		h.free = append(h.free, s)
	}
	// keep slot reuse deterministic
	sort.Slice(h.free, func(i, j int) bool { return h.free[i] > h.free[j] })
	if dead[h.entry] {
		h.entry, h.maxLevel = noSlot, -1
		for s, lv := range h.levels {
			if int(lv) > h.maxLevel || (int(lv) == h.maxLevel && lv >= 0 && h.ids[s] < h.ids[h.entry]) {
				h.entry, h.maxLevel = uint32(s), int(lv)
			}
		}
	}
}

func containsSlot(xs []uint32, s uint32) bool {
	for _, x := range xs {
		if x == s {
			return true
		}
	}
	return false
}

// Apply performs a batch of upserts and deletes (the incremental update path a real
// vector DB takes between snapshots).
func (h *Index) Apply(upserts *vec.Dataset, deletes []uint64) {
	var gone []uint64
	gone = append(gone, deletes...)
	for _, id := range upserts.IDs {
		if h.Contains(id) {
			gone = append(gone, id)
		}
	}
	h.DeleteBatch(gone)
	for i, id := range upserts.IDs {
		h.Insert(id, upserts.Row(i))
	}
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

// Result is a search hit.
type Result struct {
	ID   uint64
	Dist float32
}

// Search returns the k nearest ids using beam width ef (ef ≥ k).
func (h *Index) Search(q []float32, k, ef int) []Result {
	if h.entry == noSlot {
		return nil
	}
	if ef < k {
		ef = k
	}
	ep := []uint32{h.entry}
	for l := h.maxLevel; l > 0; l-- {
		ep = []uint32{h.searchLayer(q, ep, 1, l)[0].s}
	}
	W := h.searchLayer(q, ep, ef, 0)
	if len(W) > k {
		W = W[:k]
	}
	out := make([]Result, len(W))
	for i, c := range W {
		out[i] = Result{h.ids[c.s], c.d}
	}
	return out
}

// ---------------------------------------------------------------------------
// Snapshot interface: canonical adjacency
// ---------------------------------------------------------------------------

// Node is a node's canonical adjacency: neighbour *external ids*, sorted, per layer.
type Node struct {
	Level  int
	Layers [][]uint64
}

// Header holds graph-global state.
type Header struct {
	Dim, M, EfConstruction int
	Seed                   uint64
	EntryID                uint64
	MaxLevel               int // -1 if empty
}

// Header returns the current global state.
func (h *Index) Header() Header {
	hd := Header{Dim: h.Dim, M: h.Cfg.M, EfConstruction: h.Cfg.EfConstruction, Seed: h.Cfg.Seed, MaxLevel: h.maxLevel}
	if h.entry != noSlot {
		hd.EntryID = h.ids[h.entry]
	}
	return hd
}

// NodeAdj returns the canonical adjacency of id.
func (h *Index) NodeAdj(id uint64) (Node, bool) {
	s, ok := h.idmap[id]
	if !ok {
		return Node{}, false
	}
	n := Node{Level: int(h.levels[s]), Layers: make([][]uint64, len(h.links[s]))}
	for l, lst := range h.links[s] {
		n.Layers[l] = make([]uint64, len(lst))
		for i, x := range lst {
			n.Layers[l][i] = h.ids[x]
		}
	}
	return n, true
}

// IDs returns all live ids, sorted.
func (h *Index) IDs() []uint64 {
	out := make([]uint64, 0, len(h.idmap))
	for id := range h.idmap {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// Dirty returns ids whose adjacency changed (or which were added/removed) since the
// last ClearDirty, sorted.
func (h *Index) Dirty() []uint64 {
	out := make([]uint64, 0, len(h.dirty))
	for id := range h.dirty {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// ClearDirty resets change tracking (call after a successful snapshot commit).
func (h *Index) ClearDirty() { h.dirty = map[uint64]bool{} }

// Restore rebuilds an index from vectors + canonical adjacency without re-running
// construction. Slots are assigned in ascending id order.
func Restore(hd Header, vectors *vec.Dataset, adj map[uint64]Node) (*Index, error) {
	h := New(hd.Dim, Config{M: hd.M, EfConstruction: hd.EfConstruction, Seed: hd.Seed})
	sorted := vectors.SortByID()
	if sorted.N() != len(adj) {
		return nil, errors.New("hnsw: vectors and adjacency disagree on node count")
	}
	for i, id := range sorted.IDs {
		s := uint32(i)
		h.idmap[id] = s
		h.ids = append(h.ids, id)
		h.vecs = append(h.vecs, sorted.Row(i)...)
		n, ok := adj[id]
		if !ok {
			return nil, errors.New("hnsw: node without adjacency")
		}
		h.levels = append(h.levels, int32(n.Level))
	}
	h.links = make([][][]uint32, len(h.ids))
	for i, id := range h.ids {
		n := adj[id]
		h.links[i] = make([][]uint32, len(n.Layers))
		for l, lst := range n.Layers {
			out := make([]uint32, len(lst))
			for j, nid := range lst {
				s, ok := h.idmap[nid]
				if !ok {
					return nil, errors.New("hnsw: dangling edge")
				}
				out[j] = s
			}
			h.links[i][l] = out
		}
	}
	if hd.MaxLevel >= 0 {
		e, ok := h.idmap[hd.EntryID]
		if !ok {
			return nil, errors.New("hnsw: missing entry point")
		}
		h.entry, h.maxLevel = e, hd.MaxLevel
	}
	return h, nil
}

// MarshalFlat writes an hnswlib-like flat file: for each slot a fixed-size level-0
// block [count u32][2M links u32][vector][label u64], followed by variable upper-
// level link lists. This is what a conventional system would snapshot byte-wise, so
// the FastCDC / fixed-block baselines are measured on it.
func (h *Index) MarshalFlat() []byte {
	m0 := 2 * h.Cfg.M
	blk := 4 + 4*m0 + 4*h.Dim + 8
	out := make([]byte, 0, 32+len(h.ids)*blk)
	out = binary.LittleEndian.AppendUint32(out, uint32(len(h.ids)))
	out = binary.LittleEndian.AppendUint32(out, uint32(h.Dim))
	out = binary.LittleEndian.AppendUint32(out, uint32(h.Cfg.M))
	out = binary.LittleEndian.AppendUint32(out, h.entry)
	out = binary.LittleEndian.AppendUint32(out, uint32(int32(h.maxLevel)))
	for s := range h.ids {
		var l0 []uint32
		if h.levels[s] >= 0 {
			l0 = h.links[s][0]
		}
		out = binary.LittleEndian.AppendUint32(out, uint32(len(l0)))
		for i := 0; i < m0; i++ {
			v := uint32(0)
			if i < len(l0) {
				v = l0[i]
			}
			out = binary.LittleEndian.AppendUint32(out, v)
		}
		for _, f := range h.vecs[s*h.Dim : (s+1)*h.Dim] {
			out = binary.LittleEndian.AppendUint32(out, math.Float32bits(f))
		}
		out = binary.LittleEndian.AppendUint64(out, h.ids[s])
	}
	for s := range h.ids {
		out = binary.LittleEndian.AppendUint32(out, uint32(h.levels[s]))
		for l := 1; l < len(h.links[s]); l++ {
			lst := h.links[s][l]
			out = binary.LittleEndian.AppendUint32(out, uint32(len(lst)))
			for i := 0; i < h.Cfg.M; i++ {
				v := uint32(0)
				if i < len(lst) {
					v = lst[i]
				}
				out = binary.LittleEndian.AppendUint32(out, v)
			}
		}
	}
	return out
}

// AdjacencyBytes is the logical size of the graph (edges as u32 + per-node counts),
// used as the "uncompressed input" for dedup-ratio accounting.
func (h *Index) AdjacencyBytes() int64 {
	var n int64
	for s := range h.links {
		for _, lst := range h.links[s] {
			n += 4 + 4*int64(len(lst))
		}
	}
	return n
}
