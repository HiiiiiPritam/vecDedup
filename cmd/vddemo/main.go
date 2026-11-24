// Command vddemo serves an interactive web UI for presenting VecDedup.
//
// Every button drives the real engine (internal/repo, hnsw, quant, ...), not a mock:
//
//	go run ./cmd/vddemo            # then open http://localhost:8080
package main

import (
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/klauspost/compress/zstd"

	"vecdedup/internal/cas"
	"vecdedup/internal/hnsw"
	"vecdedup/internal/metrics"
	"vecdedup/internal/repo"
	"vecdedup/internal/synth"
	"vecdedup/internal/vec"
)

//go:embed index.html
var indexHTML []byte

type commitInfo struct {
	Hash       string  `json:"hash"`
	Short      string  `json:"short"`
	Message    string  `json:"message"`
	Time       string  `json:"time"`
	Count      int     `json:"count"`
	Mode       string  `json:"mode"`
	HasGraph   bool    `json:"hasGraph"`
	Parents    int     `json:"parents"`
	Stats      *cstats `json:"stats,omitempty"`
	RawBytes   int64   `json:"rawBytes"`
	ZstdBytes  int64   `json:"zstdBytes"`
	ChurnLabel string  `json:"churn"`
}

type cstats struct {
	Segments, SegReused, SegDelta, SegFull int
	GSegments, GReused, GDelta, GFull      int
	EdgesAdded, EdgesRemoved               int
	NewObjects, DedupObjects               int
	NewBytes                               int64
	Millis                                 float64
	Retrained                              bool
	HNSWNote                               string
}

type server struct {
	mu      sync.Mutex
	dir     string
	r       *repo.Repo
	work    *vec.Dataset
	churn   string // description of how the working set was derived
	seed    uint64
	info    map[string]*commitInfo // per snapshot hash
	exact   map[string]*vec.Dataset
	zenc    *zstd.Encoder
	queries *vec.Dataset
}

func main() {
	addr := flag.String("addr", "localhost:8080", "listen address")
	dir := flag.String("repo", filepath.Join(os.TempDir(), "vecdedup-demo"), "demo repository dir (wiped on reset)")
	results := flag.String("results", "results", "directory holding benchmark PNGs/RESULTS.md")
	flag.Parse()

	zenc, _ := zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault))
	s := &server{dir: *dir, zenc: zenc, seed: 100}
	if err := s.reset(); err != nil {
		log.Fatal(err)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(indexHTML)
	})
	mux.Handle("/results/", http.StripPrefix("/results/", http.FileServer(http.Dir(*results))))
	mux.HandleFunc("/api/state", s.wrap(s.state))
	mux.HandleFunc("/api/reset", s.wrap(s.apiReset))
	mux.HandleFunc("/api/gen", s.wrap(s.apiGen))
	mux.HandleFunc("/api/churn", s.wrap(s.apiChurn))
	mux.HandleFunc("/api/commit", s.wrap(s.apiCommit))
	mux.HandleFunc("/api/branch", s.wrap(s.apiBranch))
	mux.HandleFunc("/api/switch", s.wrap(s.apiSwitch))
	mux.HandleFunc("/api/diff", s.wrap(s.apiDiff))
	mux.HandleFunc("/api/search", s.wrap(s.apiSearch))
	mux.HandleFunc("/api/verify", s.wrap(s.apiVerify))
	mux.HandleFunc("/api/restore", s.wrap(s.apiRestore))

	fmt.Printf("VecDedup demo running -> http://%s  (repo: %s)\n", *addr, *dir)
	log.Fatal(http.ListenAndServe(*addr, mux))
}

// wrap serialises handlers (the engine is driven by one presenter) and converts
// errors to JSON.
func (s *server) wrap(f func(*http.Request) (any, error)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		s.mu.Lock()
		defer s.mu.Unlock()
		out, err := f(r)
		w.Header().Set("Content-Type", "application/json")
		if err != nil {
			w.WriteHeader(400)
			json.NewEncoder(w).Encode(map[string]string{"error": err.Error()})
			return
		}
		json.NewEncoder(w).Encode(out)
	}
}

func body(r *http.Request, v any) error {
	if r.Body == nil || r.ContentLength == 0 {
		return nil
	}
	return json.NewDecoder(r.Body).Decode(v)
}

func (s *server) reset() error {
	if s.r != nil {
		s.r.Close()
	}
	os.RemoveAll(s.dir)
	r, err := repo.Open(s.dir)
	if err != nil {
		return err
	}
	s.r = r
	s.work, s.churn, s.queries = nil, "", nil
	s.info = map[string]*commitInfo{}
	s.exact = map[string]*vec.Dataset{}
	return nil
}

func (s *server) apiReset(r *http.Request) (any, error) {
	if err := s.reset(); err != nil {
		return nil, err
	}
	return s.state(r)
}

func (s *server) workInfo() map[string]any {
	if s.work == nil {
		return nil
	}
	return map[string]any{"n": s.work.N(), "dim": s.work.Dim, "bytes": s.work.SizeBytes(), "churn": s.churn}
}

func (s *server) state(*http.Request) (any, error) {
	bs, err := s.r.Branches()
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(bs))
	for n := range bs {
		names = append(names, n)
	}
	sort.Strings(names)
	type br struct {
		Name string `json:"name"`
		Head string `json:"head"`
	}
	var brs []br
	for _, n := range names {
		brs = append(brs, br{n, bs[n].Short()})
	}

	// all reachable commits, newest first
	seen := map[cas.Hash]bool{}
	var commits []*commitInfo
	parentsOf := map[string][]string{}
	for _, n := range names {
		hs, _ := s.r.Log(bs[n].String(), 1<<20)
		for _, h := range hs {
			if seen[h] {
				continue
			}
			seen[h] = true
			sn, err := s.r.Snapshot(h)
			if err != nil {
				return nil, err
			}
			ci := s.info[h.String()]
			c := commitInfo{Hash: h.String(), Short: h.Short(), Message: sn.Message,
				Time: time.Unix(0, sn.Time).Format("15:04:05"), Count: int(sn.Count),
				Mode: sn.Mode.String(), HasGraph: sn.HasGraph(), Parents: len(sn.Parents)}
			if ci != nil {
				c.Stats, c.RawBytes, c.ZstdBytes, c.ChurnLabel = ci.Stats, ci.RawBytes, ci.ZstdBytes, ci.ChurnLabel
			}
			for _, p := range sn.Parents {
				parentsOf[c.Short] = append(parentsOf[c.Short], p.Short())
			}
			cc := c
			commits = append(commits, &cc)
		}
	}
	sort.Slice(commits, func(i, j int) bool { return commits[i].Time > commits[j].Time })

	st, err := s.r.Stats()
	if err != nil {
		return nil, err
	}
	var raw, zs int64
	for _, c := range commits {
		raw += c.RawBytes
		zs += c.ZstdBytes
	}
	return map[string]any{
		"head": s.r.HEAD(), "branches": brs, "commits": commits, "parents": parentsOf,
		"work": s.workInfo(),
		"stats": map[string]any{
			"snapshots": st.Snapshots, "objects": st.Objects, "logical": st.LogicalBytes,
			"physical": st.PhysicalBytes, "ratio": st.DedupRatio(), "raw": raw, "zstd": zs,
		},
	}, nil
}

func (s *server) apiGen(r *http.Request) (any, error) {
	var in struct{ N, Dim int }
	if err := body(r, &in); err != nil {
		return nil, err
	}
	if in.N < 100 || in.N > 200000 || in.Dim < 8 || in.Dim > 768 {
		return nil, fmt.Errorf("n must be 100..200000 and dim 8..768")
	}
	if err := s.reset(); err != nil {
		return nil, err
	}
	s.work = synth.NewGenerator(in.Dim, 100, 0.5, 1).Dataset(in.N)
	s.churn = "fresh synthetic embeddings"
	s.queries = synth.NewGenerator(in.Dim, 100, 0.5, 9).Dataset(100)
	return s.state(r)
}

func (s *server) apiChurn(r *http.Request) (any, error) {
	var in struct {
		Op   string
		Frac float64
	}
	if err := body(r, &in); err != nil {
		return nil, err
	}
	if s.work == nil {
		return nil, fmt.Errorf("generate a dataset first")
	}
	s.seed++
	g := synth.NewGenerator(s.work.Dim, 100, 0.5, s.seed)
	var nd *vec.Dataset
	var ch synth.Change
	switch in.Op {
	case "edit":
		nd, ch = synth.Edit(g, s.work, in.Frac)
	case "append":
		nd, ch = synth.Append(g, s.work, in.Frac)
	case "delete":
		nd, ch = synth.Delete(g, s.work, in.Frac)
	case "mixed":
		nd, ch = synth.Mixed(g, s.work, in.Frac)
	case "reembed":
		nd, ch = synth.Reembed(g, s.work, 1e-6)
	case "shuffle":
		nd, ch = synth.Shuffle(g, s.work)
	case "newmodel":
		nd, ch = synth.NewModel(g, s.work)
	default:
		return nil, fmt.Errorf("unknown op %q", in.Op)
	}
	s.work = nd
	s.churn = fmt.Sprintf("%s: %d upserted, %d deleted", in.Op, len(ch.Upserted), len(ch.Deleted))
	if in.Op == "edit" || in.Op == "append" || in.Op == "delete" || in.Op == "mixed" {
		s.churn = fmt.Sprintf("%s %.0f%%: %d upserted, %d deleted", in.Op, in.Frac*100, len(ch.Upserted), len(ch.Deleted))
	}
	return s.state(r)
}

// diffDatasets mirrors the CLI: upserts/deletes between two datasets (bitwise).
func diffDatasets(old, cur *vec.Dataset) (*vec.Dataset, []uint64) {
	oi := old.Index()
	ups := vec.New(cur.Dim, 0)
	seen := map[uint64]bool{}
	for i, id := range cur.IDs {
		seen[id] = true
		if j, ok := oi[id]; ok {
			a, b := old.Row(j), cur.Row(i)
			same := true
			for k := range a {
				if math.Float32bits(a[k]) != math.Float32bits(b[k]) {
					same = false
					break
				}
			}
			if same {
				continue
			}
		}
		ups.Append(id, cur.Row(i))
	}
	var dels []uint64
	for _, id := range old.IDs {
		if !seen[id] {
			dels = append(dels, id)
		}
	}
	return ups, dels
}

func (s *server) apiCommit(r *http.Request) (any, error) {
	var in struct {
		Message string
		Lossy   bool
		HNSW    bool
		Branch  string
	}
	if err := body(r, &in); err != nil {
		return nil, err
	}
	if s.work == nil {
		return nil, fmt.Errorf("generate a dataset first")
	}
	d := s.work
	opt := repo.CommitOptions{Branch: in.Branch, Message: in.Message, Lossy: repo.LossyConfig{K: 256}}
	if in.Lossy {
		opt.Mode = repo.Lossy
	}
	var g *hnsw.Index
	note := ""
	if in.HNSW {
		b := in.Branch
		if b == "" {
			b = s.r.HEAD()
		}
		head, _ := s.r.BranchHead(b)
		var parentDS *vec.Dataset
		if !head.IsZero() {
			var err error
			parentDS, g, err = s.r.Checkout(head)
			if err != nil {
				return nil, err
			}
		}
		t0 := time.Now()
		if g == nil {
			g = hnsw.New(d.Dim, hnsw.Config{M: 16, EfConstruction: 100, Seed: 42})
			for i, id := range d.IDs {
				g.Insert(id, d.Row(i))
			}
			note = fmt.Sprintf("built HNSW over %d vectors in %v", d.N(), time.Since(t0).Round(time.Millisecond))
		} else {
			ups, dels := diffDatasets(parentDS, d)
			g.ClearDirty()
			g.Apply(ups, dels)
			note = fmt.Sprintf("updated HNSW incrementally: %d upserts, %d deletes, %d nodes touched, %v",
				ups.N(), len(dels), len(g.Dirty()), time.Since(t0).Round(time.Millisecond))
		}
	}
	st, err := s.r.Commit(d, g, opt)
	if err != nil {
		return nil, err
	}
	raw := d.SizeBytes()
	if g != nil {
		raw += int64(g.Len()) * 16 * 2 * 8 // rough: edge lists as an index-file baseline
	}
	z := int64(len(s.zenc.EncodeAll(d.MarshalVDV(), nil)))
	s.info[st.Snapshot.String()] = &commitInfo{
		RawBytes: raw, ZstdBytes: z, ChurnLabel: s.churn,
		Stats: &cstats{
			Segments: st.Segments, SegReused: st.SegReused, SegDelta: st.SegDelta, SegFull: st.SegFull,
			GSegments: st.GSegments, GReused: st.GReused, GDelta: st.GDelta, GFull: st.GFull,
			EdgesAdded: st.Graph.EdgesAdded, EdgesRemoved: st.Graph.EdgesRemoved,
			NewObjects: st.NewObjects, DedupObjects: st.DedupObjects, NewBytes: st.NewBytes,
			Millis: float64(st.Duration.Microseconds()) / 1000, Retrained: st.Retrained, HNSWNote: note,
		},
	}
	if !in.Lossy {
		s.exact[st.Snapshot.String()] = d.Clone()
	}
	return s.state(r)
}

func (s *server) apiBranch(r *http.Request) (any, error) {
	var in struct{ Name, From string }
	if err := body(r, &in); err != nil {
		return nil, err
	}
	h, err := s.r.Resolve(in.From)
	if err != nil {
		return nil, err
	}
	if err := s.r.CreateBranch(in.Name, h); err != nil {
		return nil, err
	}
	return s.state(r)
}

func (s *server) apiSwitch(r *http.Request) (any, error) {
	var in struct{ Name string }
	if err := body(r, &in); err != nil {
		return nil, err
	}
	if err := s.r.Switch(in.Name); err != nil {
		return nil, err
	}
	return s.state(r)
}

func (s *server) apiDiff(r *http.Request) (any, error) {
	var in struct{ A, B string }
	if err := body(r, &in); err != nil {
		return nil, err
	}
	a, err := s.r.Resolve(in.A)
	if err != nil {
		return nil, err
	}
	b, err := s.r.Resolve(in.B)
	if err != nil {
		return nil, err
	}
	t0 := time.Now()
	d, err := s.r.Diff(a, b)
	if err != nil {
		return nil, err
	}
	return map[string]any{"a": a.Short(), "b": b.Short(), "diff": d, "ms": float64(time.Since(t0).Microseconds()) / 1000}, nil
}

func (s *server) apiSearch(r *http.Request) (any, error) {
	var in struct {
		Ref string
		K   int
		Ef  int
	}
	if err := body(r, &in); err != nil {
		return nil, err
	}
	if in.K <= 0 {
		in.K = 10
	}
	if in.Ef <= 0 {
		in.Ef = 400
	}
	h, err := s.r.Resolve(in.Ref)
	if err != nil {
		return nil, err
	}
	t0 := time.Now()
	d, g, err := s.r.Checkout(h)
	if err != nil {
		return nil, err
	}
	restore := time.Since(t0)
	if g == nil {
		return nil, fmt.Errorf("snapshot %s has no HNSW index: commit with 'HNSW index' ticked", h.Short())
	}
	if s.queries == nil || s.queries.Dim != d.Dim {
		s.queries = synth.NewGenerator(d.Dim, 100, 0.5, 9).Dataset(100)
	}
	t1 := time.Now()
	res := metrics.HNSWSearch(g, s.queries, in.K, in.Ef)
	el := time.Since(t1)
	truth := metrics.BruteForce(d, s.queries, in.K)
	return map[string]any{
		"ref": h.Short(), "restoreMs": float64(restore.Microseconds()) / 1000, "nodes": g.Len(),
		"queries": s.queries.N(), "searchMs": float64(el.Microseconds()) / 1000,
		"recall": metrics.Recall(res, truth, in.K), "k": in.K, "ef": in.Ef, "sample": res[0],
	}, nil
}

func (s *server) apiVerify(r *http.Request) (any, error) {
	bs, err := s.r.Branches()
	if err != nil {
		return nil, err
	}
	seen := map[cas.Hash]bool{}
	type row struct {
		Hash string `json:"hash"`
		OK   bool   `json:"ok"`
		Err  string `json:"err,omitempty"`
	}
	var rows []row
	t0 := time.Now()
	for _, h := range bs {
		hs, _ := s.r.Log(h.String(), 1<<20)
		for _, x := range hs {
			if seen[x] {
				continue
			}
			seen[x] = true
			if err := s.r.Verify(x); err != nil {
				rows = append(rows, row{x.Short(), false, err.Error()})
			} else {
				rows = append(rows, row{Hash: x.Short(), OK: true})
			}
		}
	}
	return map[string]any{"rows": rows, "ms": float64(time.Since(t0).Microseconds()) / 1000}, nil
}

// apiRestore checks a lossless snapshot out and compares it bit-for-bit with the
// dataset that was committed.
func (s *server) apiRestore(r *http.Request) (any, error) {
	var in struct{ Ref string }
	if err := body(r, &in); err != nil {
		return nil, err
	}
	h, err := s.r.Resolve(in.Ref)
	if err != nil {
		return nil, err
	}
	want, ok := s.exact[h.String()]
	if !ok {
		return nil, fmt.Errorf("snapshot %s is lossy (or predates this session): not bit-exact by design", h.Short())
	}
	t0 := time.Now()
	got, _, err := s.r.Checkout(h)
	if err != nil {
		return nil, err
	}
	el := time.Since(t0)
	res := map[string]any{"ref": h.Short(), "ms": float64(el.Microseconds()) / 1000, "n": got.N()}
	if err := vec.BitEqual(want, got); err != nil {
		res["identical"], res["detail"] = false, err.Error()
	} else {
		res["identical"] = true
	}
	return res, nil
}
