// Package repo implements VecDedup's versioned repository: Merkle-DAG snapshots,
// zero-copy branches, commit (full-scan and incremental), checkout, diff, verify.
//
// Object graph (every box is a content-addressed CAS object):
//
//	branch ref ──► Snapshot ──► parents (Snapshot …)
//	                 │  ├── Codebook (lossy mode, shared across snapshots)
//	                 │  ├── VecTrees[]  ──► Tree ──► vector/code segments (full | delta→base)
//	                 │  └── GraphHeader + GraphTrees[] ──► Tree ──► graph segments (full | delta→base)
//
// A Tree covers 2^TreeShift consecutive segments. Unchanged ranges keep the same Tree
// hash, so diffing two snapshots skips whole subtrees, as in Git.
package repo

import (
	"errors"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"vecdedup/internal/cas"
	"vecdedup/internal/codec"
	"vecdedup/internal/hnsw"
	"vecdedup/internal/quant"
	"vecdedup/internal/segment"
	"vecdedup/internal/vec"
)

// TreeShift: each tree object covers 256 segments (= 256K ids).
const TreeShift = 8

// Mode selects lossless (bit-exact) or lossy (residual-PQ) vector storage.
type Mode byte

const (
	Lossless Mode = 0
	Lossy    Mode = 1
)

func (m Mode) String() string {
	if m == Lossy {
		return "lossy"
	}
	return "lossless"
}

// SegRef points at one segment object.
type SegRef struct {
	Seg   uint64
	Hash  cas.Hash
	Count uint32 // rows (vectors or graph nodes) in the segment
}

// TreeRef points at one tree object.
type TreeRef struct {
	Key   uint64 // seg >> TreeShift
	Hash  cas.Hash
	Count uint64
}

// Snapshot is the commit object (its hash is the Merkle root).
type Snapshot struct {
	Parents           []cas.Hash
	Time              int64
	Message           string
	Dim               int
	Count             int
	Mode              Mode
	Codebook          cas.Hash
	LogicalVecBytes   int64
	LogicalGraphBytes int64
	VecTrees          []TreeRef
	GraphHeader       cas.Hash // zero = snapshot has no graph
	GraphTrees        []TreeRef
}

// HasGraph reports whether the snapshot carries an HNSW index.
func (s *Snapshot) HasGraph() bool { return !s.GraphHeader.IsZero() }

func writeTrees(w *codec.Writer, ts []TreeRef) {
	w.Uvarint(uint64(len(ts)))
	for _, t := range ts {
		w.Uvarint(t.Key)
		w.Raw(t.Hash[:])
		w.Uvarint(t.Count)
	}
}

func readTrees(r *codec.Reader) []TreeRef {
	n := r.Len(34)
	ts := make([]TreeRef, n)
	for i := range ts {
		ts[i].Key = r.Uvarint()
		copy(ts[i].Hash[:], r.Raw(32))
		ts[i].Count = r.Uvarint()
	}
	return ts
}

func (s *Snapshot) marshal() []byte {
	var w codec.Writer
	w.Uvarint(uint64(len(s.Parents)))
	for _, p := range s.Parents {
		w.Raw(p[:])
	}
	w.Varint(s.Time)
	w.Bytes([]byte(s.Message))
	w.Uvarint(uint64(s.Dim))
	w.Uvarint(uint64(s.Count))
	w.U8(byte(s.Mode))
	w.Raw(s.Codebook[:])
	w.Uvarint(uint64(s.LogicalVecBytes))
	w.Uvarint(uint64(s.LogicalGraphBytes))
	writeTrees(&w, s.VecTrees)
	w.Raw(s.GraphHeader[:])
	writeTrees(&w, s.GraphTrees)
	return w.B
}

func unmarshalSnapshot(b []byte) (*Snapshot, error) {
	r := &codec.Reader{B: b}
	s := &Snapshot{}
	np := r.Len(32)
	for i := 0; i < np; i++ {
		var h cas.Hash
		copy(h[:], r.Raw(32))
		s.Parents = append(s.Parents, h)
	}
	s.Time = r.Varint()
	s.Message = string(r.Bytes())
	s.Dim = int(r.Uvarint())
	s.Count = int(r.Uvarint())
	s.Mode = Mode(r.U8())
	copy(s.Codebook[:], r.Raw(32))
	s.LogicalVecBytes = int64(r.Uvarint())
	s.LogicalGraphBytes = int64(r.Uvarint())
	s.VecTrees = readTrees(r)
	copy(s.GraphHeader[:], r.Raw(32))
	s.GraphTrees = readTrees(r)
	return s, r.Err()
}

func marshalTree(refs []SegRef) []byte {
	var w codec.Writer
	w.U8('T')
	w.Uvarint(uint64(len(refs)))
	for _, r := range refs {
		w.Uvarint(r.Seg)
		w.Raw(r.Hash[:])
		w.Uvarint(uint64(r.Count))
	}
	return w.B
}

func unmarshalTree(b []byte) ([]SegRef, error) {
	r := &codec.Reader{B: b}
	if r.U8() != 'T' {
		return nil, codec.ErrCorrupt
	}
	n := r.Len(34)
	out := make([]SegRef, n)
	for i := range out {
		out[i].Seg = r.Uvarint()
		copy(out[i].Hash[:], r.Raw(32))
		out[i].Count = uint32(r.Uvarint())
	}
	return out, r.Err()
}

func marshalGraphHeader(h hnsw.Header) []byte {
	var w codec.Writer
	w.U8('H')
	w.Uvarint(uint64(h.Dim))
	w.Uvarint(uint64(h.M))
	w.Uvarint(uint64(h.EfConstruction))
	w.U64(h.Seed)
	w.U64(h.EntryID)
	w.Varint(int64(h.MaxLevel))
	return w.B
}

func unmarshalGraphHeader(b []byte) (hnsw.Header, error) {
	r := &codec.Reader{B: b}
	if r.U8() != 'H' {
		return hnsw.Header{}, codec.ErrCorrupt
	}
	h := hnsw.Header{Dim: int(r.Uvarint()), M: int(r.Uvarint()), EfConstruction: int(r.Uvarint())}
	h.Seed = r.U64()
	h.EntryID = r.U64()
	h.MaxLevel = int(r.Varint())
	return h, r.Err()
}

// ===========================================================================
// Repo
// ===========================================================================

// Repo is an open repository.
type Repo struct {
	st      *cas.Store
	writeMu sync.Mutex

	cacheMu sync.Mutex
	cache   map[cas.Hash]interface{}

	// Workers bounds encode/decode parallelism (default GOMAXPROCS).
	Workers int
}

const cacheLimit = 8192

// ErrNoSuchBranch is returned for unknown branches.
var ErrNoSuchBranch = errors.New("repo: no such branch")

// Open opens or creates a repository in dir. A fresh repository's HEAD is "main".
func Open(dir string) (*Repo, error) {
	st, err := cas.Open(dir)
	if err != nil {
		return nil, err
	}
	r := &Repo{st: st, cache: map[cas.Hash]interface{}{}, Workers: runtime.GOMAXPROCS(0)}
	if _, err := st.GetMeta("HEAD"); errors.Is(err, cas.ErrNotFound) {
		t := st.Begin()
		t.SetMeta("HEAD", []byte("main"))
		if err := t.Commit(); err != nil {
			return nil, err
		}
	}
	return r, nil
}

// Close closes the repository.
func (r *Repo) Close() error { return r.st.Close() }

// Store exposes the CAS (for stats).
func (r *Repo) Store() *cas.Store { return r.st }

// ---------------------------------------------------------------------------
// refs
// ---------------------------------------------------------------------------

// HEAD returns the current branch name.
func (r *Repo) HEAD() string {
	b, err := r.st.GetMeta("HEAD")
	if err != nil {
		return "main"
	}
	return string(b)
}

// Switch sets HEAD to an existing branch.
func (r *Repo) Switch(branch string) error {
	if _, err := r.BranchHead(branch); err != nil {
		return err
	}
	t := r.st.Begin()
	t.SetMeta("HEAD", []byte(branch))
	return t.Commit()
}

// BranchHead returns the snapshot a branch points at. A branch that exists in HEAD
// but has no commits yet returns cas.Zero.
func (r *Repo) BranchHead(branch string) (cas.Hash, error) {
	b, err := r.st.GetMeta("branch/" + branch)
	if errors.Is(err, cas.ErrNotFound) {
		if branch == r.HEAD() {
			return cas.Zero, nil
		}
		return cas.Zero, fmt.Errorf("%w: %s", ErrNoSuchBranch, branch)
	}
	if err != nil {
		return cas.Zero, err
	}
	var h cas.Hash
	copy(h[:], b)
	return h, nil
}

// Branches lists all branches and their heads.
func (r *Repo) Branches() (map[string]cas.Hash, error) {
	m, err := r.st.ListMeta("branch/")
	if err != nil {
		return nil, err
	}
	out := map[string]cas.Hash{}
	for k, v := range m {
		var h cas.Hash
		copy(h[:], v)
		out[k[len("branch/"):]] = h
	}
	return out, nil
}

// CreateBranch creates a branch pointing at `from` (a snapshot hash). This is the
// zero-copy branch: one atomic metadata write, independent of repository size.
func (r *Repo) CreateBranch(name string, from cas.Hash) error {
	if name == "" {
		return errors.New("repo: empty branch name")
	}
	if _, err := r.st.GetMeta("branch/" + name); err == nil {
		return fmt.Errorf("repo: branch %q already exists", name)
	}
	if !from.IsZero() && !r.st.Has(from) {
		return fmt.Errorf("repo: snapshot %s not found", from.Short())
	}
	t := r.st.Begin()
	t.SetMeta("branch/"+name, from[:])
	return t.Commit()
}

// DeleteBranch removes a branch ref (objects stay until a future GC).
func (r *Repo) DeleteBranch(name string) error {
	if name == r.HEAD() {
		return errors.New("repo: cannot delete the current branch")
	}
	t := r.st.Begin()
	t.DeleteMeta("branch/" + name)
	return t.Commit()
}

// Resolve turns "branch", "branch~N", or a (prefix of a) hash into a snapshot hash.
func (r *Repo) Resolve(spec string) (cas.Hash, error) {
	back := 0
	for i := len(spec) - 1; i >= 0; i-- {
		if spec[i] == '~' {
			fmt.Sscanf(spec[i+1:], "%d", &back)
			spec = spec[:i]
			break
		}
	}
	var h cas.Hash
	if spec == "" || spec == "HEAD" {
		spec = r.HEAD()
	}
	if bh, err := r.BranchHead(spec); err == nil {
		h = bh
	} else if full, err := cas.ParseHash(spec); err == nil {
		h = full
	} else {
		found := false
		for _, bh := range mustBranches(r) {
			for _, e := range r.logFrom(bh, 1<<30) {
				if len(spec) >= 4 && e.String()[:len(spec)] == spec {
					h, found = e, true
				}
			}
		}
		if !found {
			return cas.Zero, fmt.Errorf("repo: cannot resolve %q", spec)
		}
	}
	for ; back > 0 && !h.IsZero(); back-- {
		s, err := r.Snapshot(h)
		if err != nil {
			return cas.Zero, err
		}
		if len(s.Parents) == 0 {
			return cas.Zero, errors.New("repo: no such ancestor")
		}
		h = s.Parents[0]
	}
	if h.IsZero() {
		return h, errors.New("repo: branch has no commits")
	}
	return h, nil
}

func mustBranches(r *Repo) map[string]cas.Hash { b, _ := r.Branches(); return b }

func (r *Repo) logFrom(h cas.Hash, limit int) []cas.Hash {
	var out []cas.Hash
	for !h.IsZero() && len(out) < limit {
		out = append(out, h)
		s, err := r.Snapshot(h)
		if err != nil || len(s.Parents) == 0 {
			break
		}
		h = s.Parents[0]
	}
	return out
}

// Log returns up to limit snapshots following first parents from spec.
func (r *Repo) Log(spec string, limit int) ([]cas.Hash, error) {
	h, err := r.Resolve(spec)
	if err != nil {
		return nil, err
	}
	return r.logFrom(h, limit), nil
}

// ---------------------------------------------------------------------------
// object loading with cache
// ---------------------------------------------------------------------------

func (r *Repo) cached(h cas.Hash) (interface{}, bool) {
	r.cacheMu.Lock()
	defer r.cacheMu.Unlock()
	v, ok := r.cache[h]
	return v, ok
}

func (r *Repo) remember(h cas.Hash, v interface{}) {
	r.cacheMu.Lock()
	defer r.cacheMu.Unlock()
	if len(r.cache) >= cacheLimit {
		r.cache = map[cas.Hash]interface{}{}
	}
	r.cache[h] = v
}

// Snapshot loads a snapshot object.
func (r *Repo) Snapshot(h cas.Hash) (*Snapshot, error) {
	if v, ok := r.cached(h); ok {
		if s, ok := v.(*Snapshot); ok {
			return s, nil
		}
	}
	kind, b, err := r.st.Get(h)
	if err != nil {
		return nil, err
	}
	if kind != segment.KindSnapshot {
		return nil, fmt.Errorf("repo: %s is a %s, not a snapshot", h.Short(), segment.KindName(kind))
	}
	s, err := unmarshalSnapshot(b)
	if err != nil {
		return nil, err
	}
	r.remember(h, s)
	return s, nil
}

func (r *Repo) tree(h cas.Hash) ([]SegRef, error) {
	if v, ok := r.cached(h); ok {
		if t, ok := v.([]SegRef); ok {
			return t, nil
		}
	}
	_, b, err := r.st.Get(h)
	if err != nil {
		return nil, err
	}
	t, err := unmarshalTree(b)
	if err != nil {
		return nil, err
	}
	r.remember(h, t)
	return t, nil
}

// segRefs flattens a snapshot's trees into seg -> ref.
func (r *Repo) segRefs(trees []TreeRef) (map[uint64]SegRef, error) {
	out := map[uint64]SegRef{}
	for _, t := range trees {
		refs, err := r.tree(t.Hash)
		if err != nil {
			return nil, err
		}
		for _, s := range refs {
			out[s.Seg] = s
		}
	}
	return out, nil
}

// load decodes any segment/codebook object, following delta chains via the cache.
func (r *Repo) load(h cas.Hash) (interface{}, error) {
	if v, ok := r.cached(h); ok {
		return v, nil
	}
	kind, b, err := r.st.Get(h)
	if err != nil {
		return nil, err
	}
	var v interface{}
	switch kind {
	case segment.KindVFull, segment.KindVDelta:
		v, err = segment.DecodeVec(kind, b, r.load)
	case segment.KindQFull, segment.KindQDelta:
		v, err = segment.DecodeCode(kind, b, r.load)
	case segment.KindGFull, segment.KindGDelta:
		v, err = segment.DecodeGraph(kind, b, r.load)
	case segment.KindCodebook:
		var raw []byte
		if raw, err = codec.Decompress(b); err == nil {
			v, err = quant.UnmarshalCodebook(raw)
		}
	default:
		err = fmt.Errorf("repo: unexpected object kind %s", segment.KindName(kind))
	}
	if err != nil {
		return nil, fmt.Errorf("decode %s: %w", h.Short(), err)
	}
	r.remember(h, v)
	return v, nil
}

// depth returns the delta-chain depth recorded in a segment payload header.
func (r *Repo) depth(h cas.Hash) int {
	_, b, err := r.st.Get(h)
	if err != nil || len(b) < 33 {
		return segment.MaxChain
	}
	rd := &codec.Reader{B: b[32:]}
	return int(rd.Uvarint())
}

// ===========================================================================
// Commit
// ===========================================================================

// LossyConfig parameterises semantic (lossy) mode.
type LossyConfig struct {
	K       int  // coarse centroids (default 256)
	M       int  // PQ sub-spaces (default Dim/8)
	Retrain bool // train a new codebook even if the parent has one
	Seed    uint64
}

// CommitOptions controls a commit.
type CommitOptions struct {
	Branch  string // default: HEAD
	Message string
	Mode    Mode
	Lossy   LossyConfig
	Time    int64 // unix nanos; 0 = now
}

// CommitStats reports what a commit did.
type CommitStats struct {
	Snapshot                                 cas.Hash
	Segments, SegReused, SegDelta, SegFull   int
	GSegments, GReused, GDelta, GFull        int
	NewObjects, DedupObjects                 int
	NewBytes                                 int64
	Graph                                    segment.GraphDelta
	Duration                                 time.Duration
	Retrained                                bool // lossy: codebook drift detected
}

// DriftFactor: retrain the lossy codebook when reconstruction MSE on the new data
// exceeds this multiple of the training MSE.
const DriftFactor = 1.25

type segJob struct {
	seg  uint64
	vec  *segment.VecSeg
	code *segment.CodeSeg
	gr   *segment.GraphSeg
	// filled by the worker
	enc    segment.Encoded
	reused bool
	gdelta segment.GraphDelta
	err    error
}

func (r *Repo) parallel(jobs []*segJob, f func(*segJob)) {
	w := r.Workers
	if w < 1 {
		w = 1
	}
	ch := make(chan *segJob)
	var wg sync.WaitGroup
	for i := 0; i < w; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range ch {
				f(j)
			}
		}()
	}
	for _, j := range jobs {
		ch <- j
	}
	close(ch)
	wg.Wait()
}

// commitCtx carries the parent state through a commit.
type commitCtx struct {
	r          *Repo
	opt        CommitOptions
	branch     string
	parentHash cas.Hash
	parent     *Snapshot
	pVec       map[uint64]SegRef
	pGraph     map[uint64]SegRef
	codebook   *quant.Codebook
	cbHash     cas.Hash
	txn        *cas.Txn
	stats      CommitStats
}

func (r *Repo) begin(opt CommitOptions) (*commitCtx, error) {
	c := &commitCtx{r: r, opt: opt, branch: opt.Branch, pVec: map[uint64]SegRef{}, pGraph: map[uint64]SegRef{}}
	if c.branch == "" {
		c.branch = r.HEAD()
	}
	ph, err := r.BranchHead(c.branch)
	if err != nil {
		return nil, err
	}
	c.parentHash = ph
	if !ph.IsZero() {
		if c.parent, err = r.Snapshot(ph); err != nil {
			return nil, err
		}
		if c.pVec, err = r.segRefs(c.parent.VecTrees); err != nil {
			return nil, err
		}
		if c.pGraph, err = r.segRefs(c.parent.GraphTrees); err != nil {
			return nil, err
		}
	}
	c.txn = r.st.Begin()
	return c, nil
}

// setupCodebook picks (parent's) or trains a codebook for lossy mode.
func (c *commitCtx) setupCodebook(train *vec.Dataset) error {
	if c.opt.Mode != Lossy {
		return nil
	}
	if c.parent != nil && c.parent.Mode == Lossy && !c.opt.Lossy.Retrain && c.parent.Dim == train.Dim {
		v, err := c.r.load(c.parent.Codebook)
		if err != nil {
			return err
		}
		cb := v.(*quant.Codebook)
		// Drift detection: if the new data reconstructs much worse than the
		// training data did, the distribution changed (new model) -> retrain.
		if !cb.Drifted(train, DriftFactor) {
			c.codebook, c.cbHash = cb, c.parent.Codebook
			return nil
		}
		c.stats.Retrained = true
	}
	if train == nil || train.N() < 256 {
		return errors.New("repo: lossy mode needs ≥256 vectors to train a codebook")
	}
	k, m := c.opt.Lossy.K, c.opt.Lossy.M
	if k <= 0 {
		k = 256
	}
	if m <= 0 {
		m = train.Dim / 8
	}
	cb, err := quant.Train(train, k, m, c.opt.Lossy.Seed+1)
	if err != nil {
		return err
	}
	raw := cb.Marshal()
	c.cbHash = cas.Sum([]byte("codebook"), raw)
	c.codebook = cb
	c.txn.Put(c.cbHash, segment.KindCodebook, codec.Compress(raw))
	c.r.remember(c.cbHash, cb)
	return nil
}

// encodeVecJob encodes one vector segment against the parent's version.
func (c *commitCtx) encodeVecJob(j *segJob) {
	pref, hasBase := c.pVec[j.seg]
	parentLossless := c.parent != nil && c.parent.Mode == Lossless
	if c.opt.Mode == Lossless {
		h := j.vec.Hash()
		if hasBase && pref.Hash == h {
			j.enc, j.reused = segment.Encoded{Hash: h}, true
			return
		}
		var base *segment.VecSeg
		if hasBase && parentLossless {
			v, err := c.r.load(pref.Hash)
			if err != nil {
				j.err = err
				return
			}
			base = v.(*segment.VecSeg)
		}
		if base != nil {
			j.enc = segment.EncodeVec(j.vec, base, pref.Hash, c.r.depth(pref.Hash))
		} else {
			j.enc = segment.EncodeVec(j.vec, nil, cas.Zero, 0)
		}
		return
	}
	// lossy: turn the vector segment into a code segment, reusing parent codes for
	// vectors whose fingerprint did not change.
	var base *segment.CodeSeg
	if hasBase && c.parent.Mode == Lossy && c.parent.Codebook == c.cbHash {
		v, err := c.r.load(pref.Hash)
		if err != nil {
			j.err = err
			return
		}
		base = v.(*segment.CodeSeg)
	}
	out := j.code
	if out == nil {
		cs := c.codebook.CodeSize()
		out = &segment.CodeSeg{Codebook: c.cbHash, CodeSize: cs, IDs: j.vec.IDs, FPs: make([]uint64, len(j.vec.IDs)), Codes: make([]byte, len(j.vec.IDs)*cs)}
		bi := 0
		for i, id := range j.vec.IDs {
			fp := segment.Fingerprint(j.vec.Row(i))
			out.FPs[i] = fp
			for base != nil && bi < len(base.IDs) && base.IDs[bi] < id {
				bi++
			}
			if base != nil && bi < len(base.IDs) && base.IDs[bi] == id && base.FPs[bi] == fp {
				copy(out.Code(i), base.Code(bi))
				continue
			}
			c.codebook.Encode(j.vec.Row(i), out.Code(i))
		}
		j.code = out
	}
	h := out.Hash()
	if hasBase && pref.Hash == h {
		j.enc, j.reused = segment.Encoded{Hash: h}, true
		return
	}
	if base != nil {
		j.enc = segment.EncodeCode(out, base, pref.Hash, c.r.depth(pref.Hash))
	} else {
		j.enc = segment.EncodeCode(out, nil, cas.Zero, 0)
	}
}

func (c *commitCtx) encodeGraphJob(j *segJob) {
	h := j.gr.Hash()
	pref, hasBase := c.pGraph[j.seg]
	if hasBase && pref.Hash == h {
		j.enc, j.reused = segment.Encoded{Hash: h}, true
		return
	}
	var base *segment.GraphSeg
	if hasBase {
		v, err := c.r.load(pref.Hash)
		if err != nil {
			j.err = err
			return
		}
		base = v.(*segment.GraphSeg)
		j.enc = segment.EncodeGraph(j.gr, base, pref.Hash, c.r.depth(pref.Hash))
	} else {
		j.enc = segment.EncodeGraph(j.gr, nil, cas.Zero, 0)
	}
	j.gdelta = segment.DiffGraph(base, j.gr)
}

// stage puts encoded segments into the txn and returns their refs.
func (c *commitCtx) stage(jobs []*segJob, graph bool) ([]SegRef, error) {
	refs := make([]SegRef, 0, len(jobs))
	for _, j := range jobs {
		if j.err != nil {
			return nil, j.err
		}
		var count int
		switch {
		case j.code != nil:
			count = len(j.code.IDs)
		case j.vec != nil:
			count = len(j.vec.IDs)
		case j.gr != nil:
			count = len(j.gr.Nodes)
		}
		if count == 0 { // segment became empty: drop it
			continue
		}
		refs = append(refs, SegRef{Seg: j.seg, Hash: j.enc.Hash, Count: uint32(count)})
		s := &c.stats
		if graph {
			s.GSegments++
			s.Graph.NodesAdded += j.gdelta.NodesAdded
			s.Graph.NodesRemoved += j.gdelta.NodesRemoved
			s.Graph.NodesModified += j.gdelta.NodesModified
			s.Graph.EdgesAdded += j.gdelta.EdgesAdded
			s.Graph.EdgesRemoved += j.gdelta.EdgesRemoved
		} else {
			s.Segments++
		}
		if j.reused {
			if graph {
				s.GReused++
			} else {
				s.SegReused++
			}
			continue
		}
		if c.txn.Put(j.enc.Hash, j.enc.Kind, j.enc.Payload) {
			delta := j.enc.Kind == segment.KindVDelta || j.enc.Kind == segment.KindQDelta || j.enc.Kind == segment.KindGDelta
			switch {
			case graph && delta:
				s.GDelta++
			case graph:
				s.GFull++
			case delta:
				s.SegDelta++
			default:
				s.SegFull++
			}
		} else if graph {
			s.GReused++
		} else {
			s.SegReused++
		}
	}
	return refs, nil
}

// buildTrees groups refs into tree objects (unchanged trees dedupe automatically).
func (c *commitCtx) buildTrees(refs []SegRef) []TreeRef {
	sort.Slice(refs, func(a, b int) bool { return refs[a].Seg < refs[b].Seg })
	var out []TreeRef
	for i := 0; i < len(refs); {
		key := refs[i].Seg >> TreeShift
		j := i
		var cnt uint64
		for j < len(refs) && refs[j].Seg>>TreeShift == key {
			cnt += uint64(refs[j].Count)
			j++
		}
		b := marshalTree(refs[i:j])
		h := cas.Sum(b)
		c.txn.Put(h, segment.KindTree, b)
		out = append(out, TreeRef{Key: key, Hash: h, Count: cnt})
		i = j
	}
	return out
}

func (c *commitCtx) finish(dim int, vecRefs, graphRefs []SegRef, g *hnsw.Index) (CommitStats, error) {
	snap := &Snapshot{Dim: dim, Mode: c.opt.Mode, Codebook: c.cbHash, Message: c.opt.Message, Time: c.opt.Time}
	if snap.Time == 0 {
		snap.Time = time.Now().UnixNano()
	}
	if !c.parentHash.IsZero() {
		snap.Parents = []cas.Hash{c.parentHash}
	}
	snap.VecTrees = c.buildTrees(vecRefs)
	for _, t := range snap.VecTrees {
		snap.Count += int(t.Count)
	}
	snap.LogicalVecBytes = int64(snap.Count) * int64(8+4*dim)
	if g != nil {
		hb := marshalGraphHeader(g.Header())
		snap.GraphHeader = cas.Sum(hb)
		c.txn.Put(snap.GraphHeader, segment.KindGraphHeader, hb)
		snap.GraphTrees = c.buildTrees(graphRefs)
		snap.LogicalGraphBytes = g.AdjacencyBytes()
	}
	b := snap.marshal()
	h := cas.Sum(b)
	c.txn.Put(h, segment.KindSnapshot, b)
	c.txn.SetMeta("branch/"+c.branch, h[:])
	if err := c.txn.Commit(); err != nil {
		return c.stats, err
	}
	if g != nil {
		g.ClearDirty()
	}
	c.stats.Snapshot = h
	c.stats.NewObjects, c.stats.DedupObjects, c.stats.NewBytes = c.txn.NewObjects, c.txn.DedupObjects, c.txn.NewBytes
	return c.stats, nil
}

func splitVec(ds *vec.Dataset) map[uint64]*segment.VecSeg {
	sorted := ds.SortByID()
	segs := map[uint64]*segment.VecSeg{}
	for i, id := range sorted.IDs {
		k := segment.SegOf(id)
		s := segs[k]
		if s == nil {
			s = &segment.VecSeg{Dim: ds.Dim}
			segs[k] = s
		}
		if n := len(s.IDs); n > 0 && s.IDs[n-1] == id {
			return nil // duplicate id
		}
		s.IDs = append(s.IDs, id)
		s.Data = append(s.Data, sorted.Row(i)...)
	}
	return segs
}

func graphSegOf(g *hnsw.Index, ids []uint64) *segment.GraphSeg {
	gs := &segment.GraphSeg{}
	for _, id := range ids {
		n, ok := g.NodeAdj(id)
		if !ok {
			continue
		}
		gs.Nodes = append(gs.Nodes, segment.GNode{ID: id, Level: n.Level, Layers: n.Layers})
	}
	return gs
}

// Commit snapshots a complete dataset (and optionally its HNSW index) onto a branch.
// Every segment is hashed; segments identical to the parent cost nothing, changed
// ones are delta-encoded against the parent. Cost is O(dataset) for hashing plus
// O(changed) for encoding.
func (r *Repo) Commit(ds *vec.Dataset, g *hnsw.Index, opt CommitOptions) (CommitStats, error) {
	r.writeMu.Lock()
	defer r.writeMu.Unlock()
	return r.commitFull(ds, g, opt)
}

func (r *Repo) commitFull(ds *vec.Dataset, g *hnsw.Index, opt CommitOptions) (CommitStats, error) {
	t0 := time.Now()
	c, err := r.begin(opt)
	if err != nil {
		return CommitStats{}, err
	}
	if c.parent != nil && c.parent.Dim != ds.Dim {
		return CommitStats{}, fmt.Errorf("repo: dimension changed %d -> %d (use a new branch)", c.parent.Dim, ds.Dim)
	}
	if g != nil && g.Dim != ds.Dim {
		return CommitStats{}, errors.New("repo: graph and dataset dims differ")
	}
	if err := c.setupCodebook(ds); err != nil {
		return CommitStats{}, err
	}
	segs := splitVec(ds)
	if segs == nil {
		return CommitStats{}, errors.New("repo: duplicate ids in dataset")
	}
	jobs := make([]*segJob, 0, len(segs))
	for k, s := range segs {
		jobs = append(jobs, &segJob{seg: k, vec: s})
	}
	sort.Slice(jobs, func(a, b int) bool { return jobs[a].seg < jobs[b].seg })
	r.parallel(jobs, c.encodeVecJob)
	vecRefs, err := c.stage(jobs, false)
	if err != nil {
		return CommitStats{}, err
	}

	var graphRefs []SegRef
	if g != nil {
		bySeg := map[uint64][]uint64{}
		for _, id := range g.IDs() {
			bySeg[segment.SegOf(id)] = append(bySeg[segment.SegOf(id)], id)
		}
		gjobs := make([]*segJob, 0, len(bySeg))
		for k, ids := range bySeg {
			gjobs = append(gjobs, &segJob{seg: k, gr: graphSegOf(g, ids)})
		}
		// segments that disappeared entirely still count as removed nodes
		for k, ref := range c.pGraph {
			if _, ok := bySeg[k]; !ok {
				c.stats.Graph.NodesRemoved += int(ref.Count)
			}
		}
		sort.Slice(gjobs, func(a, b int) bool { return gjobs[a].seg < gjobs[b].seg })
		r.parallel(gjobs, c.encodeGraphJob)
		if graphRefs, err = c.stage(gjobs, true); err != nil {
			return CommitStats{}, err
		}
	}
	st, err := c.finish(ds.Dim, vecRefs, graphRefs, g)
	st.Duration = time.Since(t0)
	return st, err
}

// CommitChanges is the incremental path: only segments containing an upserted or
// deleted id (and graph segments containing a dirty node) are read and re-encoded.
// Everything else is carried over by hash. Cost is O(changes), independent of
// dataset size, which is what makes sub-second commits on huge indexes possible.
func (r *Repo) CommitChanges(upserts *vec.Dataset, deletes []uint64, g *hnsw.Index, opt CommitOptions) (CommitStats, error) {
	r.writeMu.Lock()
	defer r.writeMu.Unlock()
	t0 := time.Now()
	c, err := r.begin(opt)
	if err != nil {
		return CommitStats{}, err
	}
	if c.parent == nil {
		return r.commitFull(upserts, g, opt)
	}
	if upserts.Dim != c.parent.Dim {
		return CommitStats{}, errors.New("repo: dimension mismatch")
	}
	if opt.Mode != c.parent.Mode {
		return CommitStats{}, errors.New("repo: incremental commit cannot change mode; use Commit")
	}
	if opt.Mode == Lossy {
		c.codebook = nil
		v, err := r.load(c.parent.Codebook)
		if err != nil {
			return CommitStats{}, err
		}
		c.codebook, c.cbHash = v.(*quant.Codebook), c.parent.Codebook
	}

	// group changes per segment
	up := map[uint64][]int{}
	for i, id := range upserts.IDs {
		up[segment.SegOf(id)] = append(up[segment.SegOf(id)], i)
	}
	del := map[uint64]map[uint64]bool{}
	for _, id := range deletes {
		k := segment.SegOf(id)
		if del[k] == nil {
			del[k] = map[uint64]bool{}
		}
		del[k][id] = true
	}
	touched := map[uint64]bool{}
	for k := range up {
		touched[k] = true
	}
	for k := range del {
		touched[k] = true
	}

	var jobs []*segJob
	var mu sync.Mutex
	var loadErr error
	tlist := make([]*segJob, 0, len(touched))
	for k := range touched {
		tlist = append(tlist, &segJob{seg: k})
	}
	// materialise the new content of every touched segment (in parallel)
	r.parallel(tlist, func(j *segJob) {
		if opt.Mode == Lossy {
			// operate on codes directly: untouched rows keep their code + original
			// fingerprint; only upserted rows are quantized.
			var base *segment.CodeSeg
			if ref, ok := c.pVec[j.seg]; ok {
				v, err := r.load(ref.Hash)
				if err != nil {
					mu.Lock()
					loadErr = err
					mu.Unlock()
					return
				}
				base = v.(*segment.CodeSeg)
			}
			type ent struct {
				fp   uint64
				code []byte
			}
			cur := map[uint64]ent{}
			if base != nil {
				for i, id := range base.IDs {
					cur[id] = ent{base.FPs[i], base.Code(i)}
				}
			}
			for id := range del[j.seg] {
				delete(cur, id)
			}
			for _, i := range up[j.seg] {
				code := make([]byte, c.codebook.CodeSize())
				c.codebook.Encode(upserts.Row(i), code)
				cur[upserts.IDs[i]] = ent{segment.Fingerprint(upserts.Row(i)), code}
			}
			ids := make([]uint64, 0, len(cur))
			for id := range cur {
				ids = append(ids, id)
			}
			sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
			cs := &segment.CodeSeg{Codebook: c.cbHash, CodeSize: c.codebook.CodeSize(), IDs: ids}
			for _, id := range ids {
				cs.FPs = append(cs.FPs, cur[id].fp)
				cs.Codes = append(cs.Codes, cur[id].code...)
			}
			j.code = cs
			j.vec = &segment.VecSeg{Dim: upserts.Dim}
			return
		}
		cur := map[uint64][]float32{}
		if ref, ok := c.pVec[j.seg]; ok {
			vs, err := r.vectorsOfSeg(c.parent, ref.Hash)
			if err != nil {
				mu.Lock()
				loadErr = err
				mu.Unlock()
				return
			}
			for i, id := range vs.IDs {
				cur[id] = vs.Row(i)
			}
		}
		for id := range del[j.seg] {
			delete(cur, id)
		}
		for _, i := range up[j.seg] {
			cur[upserts.IDs[i]] = upserts.Row(i)
		}
		ids := make([]uint64, 0, len(cur))
		for id := range cur {
			ids = append(ids, id)
		}
		sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
		s := &segment.VecSeg{Dim: upserts.Dim, IDs: ids, Data: make([]float32, 0, len(ids)*upserts.Dim)}
		for _, id := range ids {
			s.Data = append(s.Data, cur[id]...)
		}
		j.vec = s
	})
	if loadErr != nil {
		return CommitStats{}, loadErr
	}
	jobs = tlist
	r.parallel(jobs, c.encodeVecJob)
	newRefs, err := c.stage(jobs, false)
	if err != nil {
		return CommitStats{}, err
	}
	vecRefs := make([]SegRef, 0, len(c.pVec))
	for k, ref := range c.pVec {
		if !touched[k] {
			vecRefs = append(vecRefs, ref)
			c.stats.Segments++
			c.stats.SegReused++
		}
	}
	vecRefs = append(vecRefs, newRefs...)

	var graphRefs []SegRef
	if g != nil {
		dirty := map[uint64][]uint64{}
		for _, id := range g.Dirty() {
			dirty[segment.SegOf(id)] = append(dirty[segment.SegOf(id)], id)
		}
		var gjobs []*segJob
		for k, dids := range dirty {
			members := map[uint64]bool{}
			if ref, ok := c.pGraph[k]; ok {
				v, err := r.load(ref.Hash)
				if err != nil {
					return CommitStats{}, err
				}
				for _, n := range v.(*segment.GraphSeg).Nodes {
					members[n.ID] = true
				}
			}
			for _, id := range dids {
				members[id] = true
			}
			ids := make([]uint64, 0, len(members))
			for id := range members {
				if g.Contains(id) {
					ids = append(ids, id)
				}
			}
			sort.Slice(ids, func(a, b int) bool { return ids[a] < ids[b] })
			gs := graphSegOf(g, ids)
			if len(gs.Nodes) == 0 {
				if ref, ok := c.pGraph[k]; ok {
					c.stats.Graph.NodesRemoved += int(ref.Count)
				}
			}
			gjobs = append(gjobs, &segJob{seg: k, gr: gs})
		}
		r.parallel(gjobs, c.encodeGraphJob)
		ng, err := c.stage(gjobs, true)
		if err != nil {
			return CommitStats{}, err
		}
		for k, ref := range c.pGraph {
			if _, ok := dirty[k]; !ok {
				graphRefs = append(graphRefs, ref)
				c.stats.GSegments++
				c.stats.GReused++
			}
		}
		graphRefs = append(graphRefs, ng...)
	}
	st, err := c.finish(upserts.Dim, vecRefs, graphRefs, g)
	st.Duration = time.Since(t0)
	return st, err
}

// vectorsOfSeg returns the (possibly reconstructed) vectors of a segment object.
func (r *Repo) vectorsOfSeg(snap *Snapshot, h cas.Hash) (*segment.VecSeg, error) {
	v, err := r.load(h)
	if err != nil {
		return nil, err
	}
	switch s := v.(type) {
	case *segment.VecSeg:
		return s, nil
	case *segment.CodeSeg:
		cbv, err := r.load(s.Codebook)
		if err != nil {
			return nil, err
		}
		cb := cbv.(*quant.Codebook)
		out := &segment.VecSeg{Dim: cb.Dim, IDs: s.IDs, Data: make([]float32, len(s.IDs)*cb.Dim)}
		for i := range s.IDs {
			cb.Decode(s.Code(i), out.Row(i))
		}
		return out, nil
	}
	return nil, errors.New("repo: not a vector segment")
}

// ===========================================================================
// Checkout / Verify / Diff
// ===========================================================================

// Checkout materialises a snapshot's vectors (reconstructed in lossy mode) and, if
// present, its HNSW index, which is restored directly and needs no rebuild.
func (r *Repo) Checkout(h cas.Hash) (*vec.Dataset, *hnsw.Index, error) {
	snap, err := r.Snapshot(h)
	if err != nil {
		return nil, nil, err
	}
	refs, err := r.segRefs(snap.VecTrees)
	if err != nil {
		return nil, nil, err
	}
	keys := sortedKeys(refs)
	parts := make([]*segment.VecSeg, len(keys))
	jobs := make([]*segJob, len(keys))
	for i := range keys {
		jobs[i] = &segJob{seg: uint64(i)}
	}
	r.parallel(jobs, func(j *segJob) {
		parts[j.seg], j.err = r.vectorsOfSeg(snap, refs[keys[j.seg]].Hash)
	})
	ds := vec.New(snap.Dim, snap.Count)
	for i, p := range parts {
		if jobs[i].err != nil {
			return nil, nil, jobs[i].err
		}
		ds.IDs = append(ds.IDs, p.IDs...)
		ds.Data = append(ds.Data, p.Data...)
	}
	if !snap.HasGraph() {
		return ds, nil, nil
	}
	_, hb, err := r.st.Get(snap.GraphHeader)
	if err != nil {
		return nil, nil, err
	}
	hd, err := unmarshalGraphHeader(hb)
	if err != nil {
		return nil, nil, err
	}
	grefs, err := r.segRefs(snap.GraphTrees)
	if err != nil {
		return nil, nil, err
	}
	adj := make(map[uint64]hnsw.Node, snap.Count)
	for _, k := range sortedKeys(grefs) {
		v, err := r.load(grefs[k].Hash)
		if err != nil {
			return nil, nil, err
		}
		for _, n := range v.(*segment.GraphSeg).Nodes {
			adj[n.ID] = hnsw.Node{Level: n.Level, Layers: n.Layers}
		}
	}
	g, err := hnsw.Restore(hd, ds, adj)
	return ds, g, err
}

func sortedKeys(m map[uint64]SegRef) []uint64 {
	keys := make([]uint64, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(a, b int) bool { return keys[a] < keys[b] })
	return keys
}

// Verify re-decodes every object reachable from a snapshot and checks that each
// segment's recomputed content hash matches its address (an fsck).
func (r *Repo) Verify(h cas.Hash) error {
	snap, err := r.Snapshot(h)
	if err != nil {
		return err
	}
	check := func(trees []TreeRef) error {
		for _, t := range trees {
			b := r.mustGetRaw(t.Hash)
			if cas.Sum(b) != t.Hash {
				return fmt.Errorf("tree %s: hash mismatch", t.Hash.Short())
			}
		}
		refs, err := r.segRefs(trees)
		if err != nil {
			return err
		}
		for _, ref := range refs {
			r.cacheMu.Lock()
			delete(r.cache, ref.Hash) // force a real decode
			r.cacheMu.Unlock()
			v, err := r.load(ref.Hash)
			if err != nil {
				return err
			}
			var got cas.Hash
			var n int
			switch s := v.(type) {
			case *segment.VecSeg:
				got, n = s.Hash(), len(s.IDs)
			case *segment.CodeSeg:
				got, n = s.Hash(), len(s.IDs)
			case *segment.GraphSeg:
				got, n = s.Hash(), len(s.Nodes)
			}
			if got != ref.Hash || n != int(ref.Count) {
				return fmt.Errorf("segment %d (%s): content hash/count mismatch", ref.Seg, ref.Hash.Short())
			}
		}
		return nil
	}
	if err := check(snap.VecTrees); err != nil {
		return err
	}
	if snap.HasGraph() {
		if err := check(snap.GraphTrees); err != nil {
			return err
		}
	}
	if b := r.mustGetRaw(h); cas.Sum(b) != h {
		return errors.New("snapshot: hash mismatch")
	}
	return nil
}

func (r *Repo) mustGetRaw(h cas.Hash) []byte {
	_, b, err := r.st.Get(h)
	if err != nil {
		return nil
	}
	return b
}

// DiffResult summarises the difference between two snapshots.
type DiffResult struct {
	TreesCompared, TreesSkipped int // Merkle short-circuit effectiveness
	VecAdded, VecRemoved        int
	VecChanged                  int
	Graph                       segment.GraphDelta
}

// Diff compares two snapshots, skipping identical subtrees by hash.
func (r *Repo) Diff(a, b cas.Hash) (DiffResult, error) {
	var d DiffResult
	sa, err := r.Snapshot(a)
	if err != nil {
		return d, err
	}
	sb, err := r.Snapshot(b)
	if err != nil {
		return d, err
	}
	diffTrees := func(ta, tb []TreeRef, f func(ra, rb *SegRef) error) error {
		ma, mb := map[uint64]TreeRef{}, map[uint64]TreeRef{}
		for _, t := range ta {
			ma[t.Key] = t
		}
		for _, t := range tb {
			mb[t.Key] = t
		}
		keys := map[uint64]bool{}
		for k := range ma {
			keys[k] = true
		}
		for k := range mb {
			keys[k] = true
		}
		for k := range keys {
			x, okx := ma[k]
			y, oky := mb[k]
			if okx && oky && x.Hash == y.Hash {
				d.TreesSkipped++
				continue
			}
			d.TreesCompared++
			var ra, rb []SegRef
			if okx {
				if ra, err = r.tree(x.Hash); err != nil {
					return err
				}
			}
			if oky {
				if rb, err = r.tree(y.Hash); err != nil {
					return err
				}
			}
			sm := map[uint64][2]*SegRef{}
			for i := range ra {
				e := sm[ra[i].Seg]
				e[0] = &ra[i]
				sm[ra[i].Seg] = e
			}
			for i := range rb {
				e := sm[rb[i].Seg]
				e[1] = &rb[i]
				sm[rb[i].Seg] = e
			}
			for _, e := range sm {
				if e[0] != nil && e[1] != nil && e[0].Hash == e[1].Hash {
					continue
				}
				if err := f(e[0], e[1]); err != nil {
					return err
				}
			}
		}
		return nil
	}
	err = diffTrees(sa.VecTrees, sb.VecTrees, func(ra, rb *SegRef) error {
		var va, vb *segment.VecSeg
		if ra != nil {
			if va, err = r.vectorsOfSeg(sa, ra.Hash); err != nil {
				return err
			}
		}
		if rb != nil {
			if vb, err = r.vectorsOfSeg(sb, rb.Hash); err != nil {
				return err
			}
		}
		ma := map[uint64]int{}
		if va != nil {
			for i, id := range va.IDs {
				ma[id] = i
			}
		}
		if vb != nil {
			for i, id := range vb.IDs {
				if j, ok := ma[id]; ok {
					if segment.Fingerprint(va.Row(j)) != segment.Fingerprint(vb.Row(i)) {
						d.VecChanged++
					}
					delete(ma, id)
				} else {
					d.VecAdded++
				}
			}
		}
		d.VecRemoved += len(ma)
		return nil
	})
	if err != nil {
		return d, err
	}
	err = diffTrees(sa.GraphTrees, sb.GraphTrees, func(ra, rb *SegRef) error {
		var ga, gb *segment.GraphSeg
		if ra != nil {
			v, err := r.load(ra.Hash)
			if err != nil {
				return err
			}
			ga = v.(*segment.GraphSeg)
		}
		if rb != nil {
			v, err := r.load(rb.Hash)
			if err != nil {
				return err
			}
			gb = v.(*segment.GraphSeg)
		}
		gd := segment.DiffGraph(ga, gb)
		d.Graph.NodesAdded += gd.NodesAdded
		d.Graph.NodesRemoved += gd.NodesRemoved
		d.Graph.NodesModified += gd.NodesModified
		d.Graph.EdgesAdded += gd.EdgesAdded
		d.Graph.EdgesRemoved += gd.EdgesRemoved
		return nil
	})
	return d, err
}

// RepoStats summarises space usage across all branches.
type RepoStats struct {
	Snapshots     int
	LogicalBytes  int64 // sum of uncompressed vector + graph bytes over all snapshots
	PhysicalBytes int64 // pack files + metadata on disk
	PackBytes     int64
	MetaBytes     int64
	Objects       int
}

// DedupRatio = logical / physical.
func (s RepoStats) DedupRatio() float64 {
	if s.PhysicalBytes == 0 {
		return 0
	}
	return float64(s.LogicalBytes) / float64(s.PhysicalBytes)
}

// Stats walks all snapshots reachable from any branch.
func (r *Repo) Stats() (RepoStats, error) {
	var out RepoStats
	seen := map[cas.Hash]bool{}
	br, err := r.Branches()
	if err != nil {
		return out, err
	}
	var stack []cas.Hash
	for _, h := range br {
		if !h.IsZero() {
			stack = append(stack, h)
		}
	}
	for len(stack) > 0 {
		h := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		if seen[h] {
			continue
		}
		seen[h] = true
		s, err := r.Snapshot(h)
		if err != nil {
			return out, err
		}
		out.Snapshots++
		out.LogicalBytes += s.LogicalVecBytes + s.LogicalGraphBytes
		stack = append(stack, s.Parents...)
	}
	cs, err := r.st.Stats()
	if err != nil {
		return out, err
	}
	out.PackBytes, out.MetaBytes, out.Objects = cs.PackBytes, cs.MetaBytes, cs.Objects
	out.PhysicalBytes = cs.PackBytes + cs.MetaBytes
	return out, nil
}
