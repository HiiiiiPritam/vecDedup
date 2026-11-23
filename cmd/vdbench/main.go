// Command vdbench runs the VecDedup evaluation (spec §7):
//
//  1. Vector dedup ratio per churn scenario vs. byte-level baselines
//     (full copy, zstd, fixed-block+zstd ≈ ZFS dedup, FastCDC+zstd ≈ Restic/Borg).
//  2. HNSW index snapshotting: TGIS edge deltas vs. byte-level baselines on an
//     hnswlib-style flat index file.
//  3. Lossy mode: compression vs Recall@10 / Recall@100 curve.
//  4. Latencies: commit (full-scan vs incremental), branch, checkout.
//
// Results go to results/results.json and results/RESULTS.md.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"time"

	"vecdedup/internal/cdc"
	"vecdedup/internal/codec"
	"vecdedup/internal/hnsw"
	"vecdedup/internal/metrics"
	"vecdedup/internal/repo"
	"vecdedup/internal/synth"
	"vecdedup/internal/vec"
)

type MethodResult struct {
	Method           string  `json:"method"`
	PhysicalBytes    int64   `json:"physical_bytes"`
	Ratio            float64 `json:"ratio"`             // cumulative: all snapshots
	IncrementalRatio float64 `json:"incremental_ratio"` // snapshots 2..n only
}

type ScenarioResult struct {
	Dataset      string         `json:"dataset"`
	Scenario     string         `json:"scenario"`
	Snapshots    int            `json:"snapshots"`
	LogicalBytes int64          `json:"logical_bytes"`
	Methods      []MethodResult `json:"methods"`
	Recall10     float64        `json:"lossy_recall10,omitempty"`
}

type GraphResult struct {
	Dataset        string         `json:"dataset"`
	Churn          float64        `json:"churn"`
	Nodes          int            `json:"nodes"`
	TouchedNodes   int            `json:"touched_nodes"`
	EdgesAdded     int            `json:"edges_added"`
	EdgesRemoved   int            `json:"edges_removed"`
	TotalEdges     int64          `json:"total_edges"`
	IncrementBytes map[string]int64 `json:"increment_bytes"` // physical bytes added by the 2nd snapshot
	FlatFileBytes  int64          `json:"flat_file_bytes"`
	RecallBefore   float64        `json:"recall_before"`
	RecallAfter    float64        `json:"recall_after"`
	RecallRestored float64        `json:"recall_restored"`
}

type LossyPoint struct {
	Dataset     string  `json:"dataset"`
	M           int     `json:"pq_m"`
	BytesPerVec int     `json:"bytes_per_vec"`
	Compression float64 `json:"compression"` // logical / physical for one snapshot
	Recall10    float64 `json:"recall10"`
	Recall100   float64 `json:"recall100"`
}

type LatencyResult struct {
	N              int     `json:"n"`
	Dim            int     `json:"dim"`
	Changed        int     `json:"changed"`
	FullCommitMS   float64 `json:"full_commit_ms"`
	IncrCommitMS   float64 `json:"incremental_commit_ms"`
	BranchMS       float64 `json:"branch_ms"`
	CheckoutMS     float64 `json:"checkout_ms"`
	InitialCommitMS float64 `json:"initial_commit_ms"`
}

type Results struct {
	Env       map[string]string `json:"env"`
	Scenarios []ScenarioResult  `json:"scenarios"`
	Graph     []GraphResult     `json:"graph"`
	Lossy     []LossyPoint      `json:"lossy"`
	Latency   []LatencyResult   `json:"latency"`
}

var (
	outDir  = flag.String("out", "results", "output directory")
	quick   = flag.Bool("quick", false, "small sizes (smoke test)")
	siftDir = flag.String("sift", "data/sift", "directory with sift_base_100k.fvecs and sift_query.fvecs")
	only    = flag.String("only", "", "comma list: vectors,graph,lossy,latency")
)

func logf(f string, a ...interface{}) {
	fmt.Fprintf(os.Stderr, "[%s] %s\n", time.Now().Format("15:04:05"), fmt.Sprintf(f, a...))
}

func want(part string) bool {
	if *only == "" {
		return true
	}
	for _, p := range strings.Split(*only, ",") {
		if p == part {
			return true
		}
	}
	return false
}

func loadSIFT(n, nq int) (*vec.Dataset, *vec.Dataset) {
	bf, err := os.Open(filepath.Join(*siftDir, "sift_base_100k.fvecs"))
	if err != nil {
		return nil, nil
	}
	defer bf.Close()
	qf, err := os.Open(filepath.Join(*siftDir, "sift_query.fvecs"))
	if err != nil {
		return nil, nil
	}
	defer qf.Close()
	b, err := vec.ReadFvecs(bf, n)
	if err != nil {
		return nil, nil
	}
	q, err := vec.ReadFvecs(qf, nq)
	if err != nil {
		return nil, nil
	}
	return b, q
}

// ---------------------------------------------------------------------------
// 1. Vector dedup scenarios
// ---------------------------------------------------------------------------

type step func(*synth.Generator, *vec.Dataset) *vec.Dataset

func runScenario(dsName, name string, base *vec.Dataset, g *synth.Generator, f step, steps int, queries *vec.Dataset) ScenarioResult {
	snaps := []*vec.Dataset{base}
	cur := base
	for i := 0; i < steps; i++ {
		cur = f(g, cur)
		snaps = append(snaps, cur)
	}
	res := ScenarioResult{Dataset: dsName, Scenario: name, Snapshots: len(snaps)}
	files := make([][]byte, len(snaps))
	var logicalFirst int64
	for i, s := range snaps {
		files[i] = s.MarshalVDV()
		res.LogicalBytes += int64(len(files[i]))
		if i == 0 {
			logicalFirst = int64(len(files[i]))
		}
	}
	incLogical := res.LogicalBytes - logicalFirst
	add := func(m string, phys, physFirst int64) {
		mr := MethodResult{Method: m, PhysicalBytes: phys, Ratio: float64(res.LogicalBytes) / float64(phys)}
		if phys-physFirst > 0 {
			mr.IncrementalRatio = float64(incLogical) / float64(phys-physFirst)
		}
		res.Methods = append(res.Methods, mr)
	}

	// full copies
	add("full-copy", res.LogicalBytes, logicalFirst)
	// zstd per snapshot file
	var z, zFirst int64
	for i, f := range files {
		n := int64(len(codec.Compress(f)))
		z += n
		if i == 0 {
			zFirst = n
		}
	}
	add("zstd", z, zFirst)
	// fixed blocks (ZFS-style dedup, recordsize 128K) + zstd
	fb := cdc.NewDedup(true)
	var fbFirst int64
	for i, f := range files {
		fb.AddFixed(f, 128<<10)
		if i == 0 {
			fbFirst = fb.Physical
		}
	}
	add("fixed128K+zstd", fb.Physical, fbFirst)
	// FastCDC 8K + zstd
	cd := cdc.NewDedup(true)
	var cdFirst int64
	for i, f := range files {
		cd.AddCDC(f, cdc.DefaultParams())
		if i == 0 {
			cdFirst = cd.Physical
		}
	}
	add("fastcdc8K+zstd", cd.Physical, cdFirst)

	// VecDedup lossless + lossy
	for _, mode := range []repo.Mode{repo.Lossless, repo.Lossy} {
		dir, _ := os.MkdirTemp("", "vdbench-*")
		r, err := repo.Open(dir)
		if err != nil {
			panic(err)
		}
		var first int64
		var last *vec.Dataset
		var lastHash = [32]byte{}
		for i, s := range snaps {
			st, err := r.Commit(s, nil, repo.CommitOptions{Mode: mode, Lossy: repo.LossyConfig{K: 256, M: s.Dim / 8}, Time: int64(i + 1)})
			if err != nil {
				panic(err)
			}
			if i == 0 {
				ps, _ := r.Stats()
				first = ps.PhysicalBytes
			}
			last, lastHash = s, st.Snapshot
		}
		ps, _ := r.Stats()
		name := "vecdedup-lossless"
		if mode == repo.Lossy {
			name = fmt.Sprintf("vecdedup-lossy(M=D/8)")
			if queries != nil {
				got, _, err := r.Checkout(lastHash)
				if err == nil {
					// Queries must come from the distribution of the snapshot being
					// measured (after a model change, old queries are meaningless).
					q := inDistributionQueries(last, queries.N(), 77)
					res.Recall10 = metrics.Recall(metrics.BruteForce(got, q, 10), metrics.BruteForce(last, q, 10), 10)
				}
			}
		}
		add(name, ps.PhysicalBytes, first)
		r.Close()
		os.RemoveAll(dir)
	}
	return res
}

// inDistributionQueries builds held-out-style queries by blending random pairs of
// rows (0.7·a + 0.3·b), so queries follow the dataset's own distribution but are
// not identical to any stored vector.
func inDistributionQueries(d *vec.Dataset, n int, seed uint64) *vec.Dataset {
	g := synth.NewGenerator(1, 1, 0.5, seed)
	rng := g.Rand()
	q := vec.New(d.Dim, n)
	v := make([]float32, d.Dim)
	for i := 0; i < n; i++ {
		a, b := d.Row(rng.IntN(d.N())), d.Row(rng.IntN(d.N()))
		for k := range v {
			v[k] = 0.7*a[k] + 0.3*b[k]
		}
		q.Append(uint64(i), v)
	}
	return q
}

func vectorScenarios(res *Results) {
	type dsSpec struct {
		name    string
		base    *vec.Dataset
		g       *synth.Generator
		queries *vec.Dataset
	}
	var dss []dsSpec
	n, dim := 50000, 384
	if *quick {
		n, dim = 5000, 64
	}
	g := synth.NewGenerator(dim, 100, 0.5, 11)
	dss = append(dss, dsSpec{fmt.Sprintf("synthetic-%dx%d", n, dim), g.Dataset(n), g, g.Dataset(100)})
	if !*quick {
		if b, q := loadSIFT(100000, 100); b != nil {
			dss = append(dss, dsSpec{"sift-100kx128", b, synth.NewGenerator(128, 100, 0.5, 12), q})
		}
	}
	steps := 3
	for _, ds := range dss {
		scen := []struct {
			name string
			f    step
		}{
			{"edit-1%", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.Edit(g, d, 0.01); return x }},
			{"edit-5%", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.Edit(g, d, 0.05); return x }},
			{"edit-10%", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.Edit(g, d, 0.10); return x }},
			{"edit-25%", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.Edit(g, d, 0.25); return x }},
			{"mixed-5%", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.Mixed(g, d, 0.05); return x }},
			{"reembed-noise", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.Reembed(g, d, 1e-6); return x }},
			{"compaction-shuffle", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset {
				x, _ := synth.Delete(g, d, 0.01)
				y, _ := synth.Shuffle(g, x)
				return y
			}},
			{"new-model", func(g *synth.Generator, d *vec.Dataset) *vec.Dataset { x, _ := synth.NewModel(g, d); return x }},
		}
		for _, s := range scen {
			t0 := time.Now()
			r := runScenario(ds.name, s.name, ds.base, ds.g, s.f, steps, ds.queries)
			res.Scenarios = append(res.Scenarios, r)
			logf("vectors %-18s %-20s done in %v", ds.name, s.name, time.Since(t0).Round(time.Millisecond))
		}
	}
}

// ---------------------------------------------------------------------------
// 2. HNSW graph snapshots
// ---------------------------------------------------------------------------

func graphBench(res *Results) {
	var base, queries *vec.Dataset
	name := ""
	n := 50000
	if *quick {
		n = 4000
	}
	if b, q := loadSIFT(n, 200); b != nil && !*quick {
		base, queries, name = b, q, fmt.Sprintf("sift-%dx128", n)
	} else {
		g := synth.NewGenerator(128, 100, 0.5, 21)
		base, queries, name = g.Dataset(n), g.Dataset(200), fmt.Sprintf("synthetic-%dx128", n)
	}
	logf("graph: building HNSW over %s", name)
	t0 := time.Now()
	h0 := hnsw.New(base.Dim, hnsw.Config{M: 16, EfConstruction: 100, Seed: 3})
	for i, id := range base.IDs {
		h0.Insert(id, base.Row(i))
	}
	logf("graph: built in %v", time.Since(t0).Round(time.Millisecond))
	truth0 := metrics.BruteForce(base, queries, 10)
	recall0 := metrics.Recall(metrics.HNSWSearch(h0, queries, 10, 64), truth0, 10)
	flat0 := h0.MarshalFlat()
	adj0 := map[uint64]hnsw.Node{}
	for _, id := range h0.IDs() {
		adj0[id], _ = h0.NodeAdj(id)
	}

	for _, churn := range []float64{0.01, 0.05, 0.10} {
		// fresh copy of the base index per churn level
		h, _ := hnsw.Restore(h0.Header(), base, adj0)
		gen := synth.NewGenerator(base.Dim, 100, 0.5, uint64(churn*1000)+5)
		// For SIFT, generated vectors would be off-distribution; reuse real vectors
		// from the tail of the file as "new documents" by sampling and jittering.
		d2, ch := mixedLike(gen, base, churn)
		idx := d2.Index()
		ups := vec.New(base.Dim, len(ch.Upserted))
		for _, id := range ch.Upserted {
			ups.Append(id, d2.Row(idx[id]))
		}

		dir, _ := os.MkdirTemp("", "vdgraph-*")
		r, _ := repo.Open(dir)
		if _, err := r.Commit(base, h, repo.CommitOptions{Time: 1}); err != nil {
			panic(err)
		}
		ps0, _ := r.Stats()
		h.ClearDirty()
		h.Apply(ups, ch.Deleted)
		touched := len(h.Dirty())
		st, err := r.CommitChanges(ups, ch.Deleted, h, repo.CommitOptions{Time: 2})
		if err != nil {
			panic(err)
		}
		ps1, _ := r.Stats()
		truth1 := metrics.BruteForce(d2, queries, 10)
		recall1 := metrics.Recall(metrics.HNSWSearch(h, queries, 10, 64), truth1, 10)
		_, rg, err := r.Checkout(st.Snapshot)
		if err != nil {
			panic(err)
		}
		recallR := metrics.Recall(metrics.HNSWSearch(rg, queries, 10, 64), truth1, 10)
		r.Close()
		os.RemoveAll(dir)

		flat1 := h.MarshalFlat()
		gr := GraphResult{Dataset: name, Churn: churn, Nodes: h.Len(), TouchedNodes: touched,
			EdgesAdded: st.Graph.EdgesAdded, EdgesRemoved: st.Graph.EdgesRemoved,
			TotalEdges: (h.AdjacencyBytes() - int64(h.Len())*4) / 4, FlatFileBytes: int64(len(flat1)),
			IncrementBytes: map[string]int64{}, RecallBefore: recall0, RecallAfter: recall1, RecallRestored: recallR}
		gr.IncrementBytes["full-copy"] = int64(len(flat1))
		gr.IncrementBytes["zstd"] = int64(len(codec.Compress(flat1)))
		fb := cdc.NewDedup(true)
		fb.AddFixed(flat0, 128<<10)
		p := fb.Physical
		fb.AddFixed(flat1, 128<<10)
		gr.IncrementBytes["fixed128K+zstd"] = fb.Physical - p
		cd := cdc.NewDedup(true)
		cd.AddCDC(flat0, cdc.DefaultParams())
		p = cd.Physical
		cd.AddCDC(flat1, cdc.DefaultParams())
		gr.IncrementBytes["fastcdc8K+zstd"] = cd.Physical - p
		gr.IncrementBytes["vecdedup(vectors+TGIS)"] = ps1.PhysicalBytes - ps0.PhysicalBytes
		res.Graph = append(res.Graph, gr)
		logf("graph churn %.0f%%: touched %d/%d nodes, edges +%d -%d, vecdedup +%d B vs fastcdc +%d B",
			churn*100, touched, h.Len(), gr.EdgesAdded, gr.EdgesRemoved, gr.IncrementBytes["vecdedup(vectors+TGIS)"], gr.IncrementBytes["fastcdc8K+zstd"])
	}
}

// mixedLike applies edit/append/delete churn where new vectors are jittered copies
// of existing vectors (keeps them on the dataset's distribution, e.g. for SIFT).
func mixedLike(g *synth.Generator, d *vec.Dataset, frac float64) (*vec.Dataset, synth.Change) {
	rng := g.Rand()
	out := d.Clone()
	var ch synth.Change
	n := d.N()
	jitter := func(src []float32) []float32 {
		v := make([]float32, len(src))
		other := d.Row(rng.IntN(n))
		for i := range v {
			v[i] = 0.7*src[i] + 0.3*other[i]
		}
		return v
	}
	edits := int(frac * 0.5 * float64(n))
	for _, i := range rng.Perm(n)[:edits] {
		copy(out.Row(i), jitter(d.Row(i)))
		ch.Upserted = append(ch.Upserted, out.IDs[i])
	}
	next := uint64(0)
	for _, id := range d.IDs {
		if id >= next {
			next = id + 1
		}
	}
	for i := 0; i < int(frac*0.3*float64(n)); i++ {
		out.Append(next, jitter(d.Row(rng.IntN(n))))
		ch.Upserted = append(ch.Upserted, next)
		next++
	}
	edited := map[uint64]bool{}
	for _, id := range ch.Upserted {
		edited[id] = true
	}
	dels := map[uint64]bool{}
	for _, i := range rng.Perm(n)[:int(frac*0.2*float64(n))] {
		if !edited[d.IDs[i]] {
			dels[d.IDs[i]] = true
			ch.Deleted = append(ch.Deleted, d.IDs[i])
		}
	}
	final := vec.New(d.Dim, out.N())
	for i, id := range out.IDs {
		if !dels[id] {
			final.Append(id, out.Row(i))
		}
	}
	sort.Slice(ch.Deleted, func(a, b int) bool { return ch.Deleted[a] < ch.Deleted[b] })
	return final, ch
}

// ---------------------------------------------------------------------------
// 3. Lossy curve
// ---------------------------------------------------------------------------

func lossyCurve(res *Results) {
	type ds struct {
		name    string
		base, q *vec.Dataset
	}
	var list []ds
	n, dim := 50000, 384
	if *quick {
		n, dim = 4000, 64
	}
	g := synth.NewGenerator(dim, 100, 0.5, 31)
	list = append(list, ds{fmt.Sprintf("synthetic-%dx%d", n, dim), g.Dataset(n), g.Dataset(200)})
	if !*quick {
		if b, q := loadSIFT(100000, 200); b != nil {
			list = append(list, ds{"sift-100kx128", b, q})
		}
	}
	for _, d := range list {
		t10 := metrics.BruteForce(d.base, d.q, 10)
		t100 := metrics.BruteForce(d.base, d.q, 100)
		for _, div := range []int{16, 8, 4, 2} {
			m := d.base.Dim / div
			if m < 1 {
				continue
			}
			dir, _ := os.MkdirTemp("", "vdlossy-*")
			r, _ := repo.Open(dir)
			st, err := r.Commit(d.base, nil, repo.CommitOptions{Mode: repo.Lossy, Lossy: repo.LossyConfig{K: 256, M: m}})
			if err != nil {
				panic(err)
			}
			ps, _ := r.Stats()
			got, _, _ := r.Checkout(st.Snapshot)
			p := LossyPoint{Dataset: d.name, M: m, BytesPerVec: 2 + m,
				Compression: float64(d.base.SizeBytes()) / float64(ps.PhysicalBytes),
				Recall10:    metrics.Recall(metrics.BruteForce(got, d.q, 10), t10, 10),
				Recall100:   metrics.Recall(metrics.BruteForce(got, d.q, 100), t100, 100)}
			res.Lossy = append(res.Lossy, p)
			logf("lossy %s M=%d: %.1fx, recall@10=%.3f recall@100=%.3f", d.name, m, p.Compression, p.Recall10, p.Recall100)
			r.Close()
			os.RemoveAll(dir)
		}
	}
}

// ---------------------------------------------------------------------------
// 4. Latency
// ---------------------------------------------------------------------------

func latency(res *Results) {
	sizes := []int{100000, 1000000}
	if *quick {
		sizes = []int{20000}
	}
	dim := 128
	for _, n := range sizes {
		g := synth.NewGenerator(dim, 100, 0.5, 41)
		base := g.Dataset(n)
		changed := 1000
		d2, _ := synth.Edit(g, base, float64(changed)/float64(n))
		idx := d2.Index()
		bi := base.Index()
		ups := vec.New(dim, changed)
		for _, id := range d2.IDs {
			a, b := d2.Row(idx[id]), base.Row(bi[id])
			for k := range a {
				if a[k] != b[k] {
					ups.Append(id, a)
					break
				}
			}
		}
		lr := LatencyResult{N: n, Dim: dim, Changed: ups.N()}

		run := func(incremental bool) (float64, float64, float64, float64) {
			dir, _ := os.MkdirTemp("", "vdlat-*")
			defer os.RemoveAll(dir)
			r, _ := repo.Open(dir)
			defer r.Close()
			t0 := time.Now()
			if _, err := r.Commit(base, nil, repo.CommitOptions{}); err != nil {
				panic(err)
			}
			initial := ms(time.Since(t0))
			t0 = time.Now()
			var st repo.CommitStats
			var err error
			if incremental {
				st, err = r.CommitChanges(ups, nil, nil, repo.CommitOptions{})
			} else {
				st, err = r.Commit(d2, nil, repo.CommitOptions{})
			}
			if err != nil {
				panic(err)
			}
			commit := ms(time.Since(t0))
			const reps = 50
			t0 = time.Now()
			for i := 0; i < reps; i++ {
				if err := r.CreateBranch(fmt.Sprintf("b%d", i), st.Snapshot); err != nil {
					panic(err)
				}
			}
			branch := ms(time.Since(t0)) / reps
			t0 = time.Now()
			if _, _, err := r.Checkout(st.Snapshot); err != nil {
				panic(err)
			}
			return initial, commit, branch, ms(time.Since(t0))
		}
		lr.InitialCommitMS, lr.FullCommitMS, _, _ = run(false)
		_, lr.IncrCommitMS, lr.BranchMS, lr.CheckoutMS = run(true)
		res.Latency = append(res.Latency, lr)
		logf("latency N=%d: initial %.0fms, full-scan commit %.1fms, incremental %.1fms, branch %.2fms, checkout %.0fms",
			n, lr.InitialCommitMS, lr.FullCommitMS, lr.IncrCommitMS, lr.BranchMS, lr.CheckoutMS)
	}
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// ---------------------------------------------------------------------------
// report
// ---------------------------------------------------------------------------

func human(n int64) string {
	f := float64(n)
	switch {
	case f >= 1<<30:
		return fmt.Sprintf("%.2f GiB", f/(1<<30))
	case f >= 1<<20:
		return fmt.Sprintf("%.1f MiB", f/(1<<20))
	case f >= 1<<10:
		return fmt.Sprintf("%.1f KiB", f/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func writeMarkdown(res *Results, path string) error {
	var b strings.Builder
	fmt.Fprintf(&b, "# VecDedup benchmark results\n\nEnvironment: %s, %s CPUs, %s. Generated %s.\n\n",
		res.Env["os"], res.Env["cpus"], res.Env["go"], res.Env["time"])
	if len(res.Scenarios) > 0 {
		b.WriteString("## 1. Vector snapshots: dedup ratio (base + 3 churned snapshots)\n\n")
		b.WriteString("Cumulative ratio = all snapshot bytes / physical bytes. *Incr.* = ratio for snapshots 2–4 only (what each new snapshot costs).\n\n")
		methods := []string{}
		for _, m := range res.Scenarios[0].Methods {
			methods = append(methods, m.Method)
		}
		for _, dsn := range uniqueDatasets(res.Scenarios) {
			fmt.Fprintf(&b, "### %s\n\n| scenario |", dsn)
			for _, m := range methods {
				fmt.Fprintf(&b, " %s |", m)
			}
			b.WriteString(" lossy R@10 |\n|---|")
			for range methods {
				b.WriteString("---|")
			}
			b.WriteString("---|\n")
			for _, s := range res.Scenarios {
				if s.Dataset != dsn {
					continue
				}
				fmt.Fprintf(&b, "| %s |", s.Scenario)
				for _, m := range s.Methods {
					inc := "–"
					if m.IncrementalRatio > 0 {
						inc = fmt.Sprintf("%.1f", m.IncrementalRatio)
					}
					fmt.Fprintf(&b, " %.2f× (incr. %s×) |", m.Ratio, inc)
				}
				fmt.Fprintf(&b, " %.3f |\n", s.Recall10)
			}
			b.WriteString("\n")
		}
	}
	if len(res.Graph) > 0 {
		b.WriteString("## 2. HNSW index snapshots (TGIS)\n\nBytes added by the snapshot taken *after* an incremental index update. Baselines snapshot the hnswlib-style flat index file (vectors + links).\n\n")
		b.WriteString("| dataset | churn | touched nodes | ΔE+ | ΔE− | full copy | zstd | fixed128K+zstd | FastCDC+zstd | **VecDedup** | recall before / after / restored |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
		for _, g := range res.Graph {
			fmt.Fprintf(&b, "| %s | %.0f%% | %d / %d (%.0f%%) | %d | %d | %s | %s | %s | %s | **%s** | %.3f / %.3f / %.3f |\n",
				g.Dataset, g.Churn*100, g.TouchedNodes, g.Nodes, 100*float64(g.TouchedNodes)/float64(g.Nodes), g.EdgesAdded, g.EdgesRemoved,
				human(g.IncrementBytes["full-copy"]), human(g.IncrementBytes["zstd"]), human(g.IncrementBytes["fixed128K+zstd"]),
				human(g.IncrementBytes["fastcdc8K+zstd"]), human(g.IncrementBytes["vecdedup(vectors+TGIS)"]), g.RecallBefore, g.RecallAfter, g.RecallRestored)
		}
		b.WriteString("\n")
	}
	if len(res.Lossy) > 0 {
		b.WriteString("## 3. Lossy (semantic) mode: compression vs recall\n\nExact search over reconstructed vectors vs. ground truth on originals. Single snapshot; includes codebook.\n\n| dataset | PQ M | bytes/vector | compression | Recall@10 | Recall@100 |\n|---|---|---|---|---|---|\n")
		for _, p := range res.Lossy {
			fmt.Fprintf(&b, "| %s | %d | %d | %.1f× | %.3f | %.3f |\n", p.Dataset, p.M, p.BytesPerVec, p.Compression, p.Recall10, p.Recall100)
		}
		b.WriteString("\n")
	}
	if len(res.Latency) > 0 {
		b.WriteString("## 4. Latency\n\n| vectors | dim | changed | initial commit | full-scan commit | incremental commit | branch | checkout |\n|---|---|---|---|---|---|---|---|\n")
		for _, l := range res.Latency {
			fmt.Fprintf(&b, "| %d | %d | %d | %.0f ms | %.1f ms | **%.1f ms** | **%.2f ms** | %.0f ms |\n", l.N, l.Dim, l.Changed, l.InitialCommitMS, l.FullCommitMS, l.IncrCommitMS, l.BranchMS, l.CheckoutMS)
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func uniqueDatasets(s []ScenarioResult) []string {
	var out []string
	seen := map[string]bool{}
	for _, x := range s {
		if !seen[x.Dataset] {
			seen[x.Dataset] = true
			out = append(out, x.Dataset)
		}
	}
	return out
}

func main() {
	flag.Parse()
	os.MkdirAll(*outDir, 0o755)
	res := &Results{}
	if *only != "" { // re-running some parts: keep the others from the previous run
		if b, err := os.ReadFile(filepath.Join(*outDir, "results.json")); err == nil {
			json.Unmarshal(b, res)
		}
	}
	res.Env = map[string]string{
		"os": runtime.GOOS + "/" + runtime.GOARCH, "cpus": fmt.Sprint(runtime.NumCPU()),
		"go": runtime.Version(), "time": time.Now().Format(time.RFC3339), "quick": fmt.Sprint(*quick)}
	if want("vectors") {
		res.Scenarios = nil
		vectorScenarios(res)
	}
	if want("graph") {
		res.Graph = nil
		graphBench(res)
	}
	if want("lossy") {
		res.Lossy = nil
		lossyCurve(res)
	}
	if want("latency") {
		res.Latency = nil
		latency(res)
	}
	js, _ := json.MarshalIndent(res, "", "  ")
	if err := os.WriteFile(filepath.Join(*outDir, "results.json"), js, 0o644); err != nil {
		panic(err)
	}
	if err := writeMarkdown(res, filepath.Join(*outDir, "RESULTS.md")); err != nil {
		panic(err)
	}
	logf("wrote %s", filepath.Join(*outDir, "RESULTS.md"))
}
