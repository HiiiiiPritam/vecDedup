// Command vecdedup is the CLI for the VecDedup snapshot engine.
//
//	vecdedup [-C repo] <command> [flags]
//
// Run `vecdedup help` for the command list.
package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"vecdedup/internal/cas"
	"vecdedup/internal/hnsw"
	"vecdedup/internal/metrics"
	"vecdedup/internal/repo"
	"vecdedup/internal/synth"
	"vecdedup/internal/vec"
)

const usage = `vecdedup — semantic-aware snapshot & dedup engine for vector data

Usage: vecdedup [-C <repo-dir>] <command> [flags]

Repository commands:
  init                         create a repository (also implied by any write)
  commit <data.vdv|.fvecs>     snapshot a dataset onto the current branch
        -m <msg>  --branch <b>  --lossy  --pq-m <M>  --k <K>  --retrain
        --hnsw    maintain an HNSW index incrementally and snapshot it too
  log [<ref>]                  show history
  branch                       list branches
  branch <name> [<ref>]        create a zero-copy branch at <ref> (default HEAD)
  switch <branch>              change the current branch
  checkout <ref> -o <out.vdv>  materialise a snapshot's vectors
  diff <refA> <refB>           vector + graph (ΔE+, ΔE−) differences
  search <ref> --query <q.vdv> [--k 10]   query a snapshot's restored HNSW index
  verify [<ref>]               re-decode every object and check content hashes
  stats                        logical vs physical size, dedup ratio

Data tools (for demos):
  gen   --n N --dim D --out f.vdv [--seed S]
  churn --in f.vdv --op edit|append|delete|mixed|reembed|shuffle|newmodel
        --frac F --out g.vdv [--noise 1e-6] [--seed S]

<ref> = branch | branch~N | snapshot hash prefix
`

func main() {
	args := os.Args[1:]
	dir := ".vecdedup"
	if len(args) >= 2 && args[0] == "-C" {
		dir, args = args[1], args[2:]
	}
	if len(args) == 0 || args[0] == "help" || args[0] == "-h" || args[0] == "--help" {
		fmt.Print(usage)
		return
	}
	if err := run(dir, args[0], args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func human(n int64) string {
	f := float64(n)
	for _, u := range []string{"B", "KiB", "MiB", "GiB"} {
		if math.Abs(f) < 1024 || u == "GiB" {
			if u == "B" {
				return fmt.Sprintf("%d B", n)
			}
			return fmt.Sprintf("%.2f %s", f, u)
		}
		f /= 1024
	}
	return ""
}

// parse lets flags and positional args be mixed freely.
func parse(fs *flag.FlagSet, args []string) []string {
	var pos []string
	for len(args) > 0 {
		fs.Parse(args)
		args = fs.Args()
		if len(args) > 0 {
			pos = append(pos, args[0])
			args = args[1:]
		}
	}
	return pos
}

func run(dir, cmd string, args []string) error {
	switch cmd {
	case "gen":
		return cmdGen(args)
	case "churn":
		return cmdChurn(args)
	}
	r, err := repo.Open(dir)
	if err != nil {
		return err
	}
	defer r.Close()
	switch cmd {
	case "init":
		fmt.Printf("initialised repository in %s (HEAD -> %s)\n", dir, r.HEAD())
		return nil
	case "commit":
		return cmdCommit(r, args)
	case "log":
		return cmdLog(r, args)
	case "branch":
		return cmdBranch(r, args)
	case "switch":
		if len(args) != 1 {
			return errors.New("usage: switch <branch>")
		}
		if err := r.Switch(args[0]); err != nil {
			return err
		}
		fmt.Println("switched to", args[0])
		return nil
	case "checkout":
		return cmdCheckout(r, args)
	case "diff":
		return cmdDiff(r, args)
	case "search":
		return cmdSearch(r, args)
	case "verify":
		return cmdVerify(r, args)
	case "stats":
		return cmdStats(r)
	}
	return fmt.Errorf("unknown command %q (try `vecdedup help`)", cmd)
}

func cmdGen(args []string) error {
	fs := flag.NewFlagSet("gen", flag.ExitOnError)
	n := fs.Int("n", 100000, "vectors")
	dim := fs.Int("dim", 384, "dimension")
	out := fs.String("out", "base.vdv", "output file")
	seed := fs.Uint64("seed", 1, "seed")
	parse(fs, args)
	d := synth.NewGenerator(*dim, 100, 0.5, *seed).Dataset(*n)
	if err := d.WriteFile(*out); err != nil {
		return err
	}
	fmt.Printf("wrote %d x %d vectors to %s (%s)\n", *n, *dim, *out, human(d.SizeBytes()))
	return nil
}

func cmdChurn(args []string) error {
	fs := flag.NewFlagSet("churn", flag.ExitOnError)
	in := fs.String("in", "", "input .vdv")
	op := fs.String("op", "edit", "edit|append|delete|mixed|reembed|shuffle|newmodel")
	frac := fs.Float64("frac", 0.05, "fraction of rows affected")
	noise := fs.Float64("noise", 1e-6, "relative noise for reembed")
	out := fs.String("out", "", "output .vdv")
	seed := fs.Uint64("seed", 2, "seed")
	parse(fs, args)
	if *in == "" || *out == "" {
		return errors.New("churn needs --in and --out")
	}
	d, err := vec.ReadFile(*in)
	if err != nil {
		return err
	}
	g := synth.NewGenerator(d.Dim, 100, 0.5, *seed)
	var nd *vec.Dataset
	var ch synth.Change
	switch *op {
	case "edit":
		nd, ch = synth.Edit(g, d, *frac)
	case "append":
		nd, ch = synth.Append(g, d, *frac)
	case "delete":
		nd, ch = synth.Delete(g, d, *frac)
	case "mixed":
		nd, ch = synth.Mixed(g, d, *frac)
	case "reembed":
		nd, ch = synth.Reembed(g, d, *noise)
	case "shuffle":
		nd, ch = synth.Shuffle(g, d)
	case "newmodel":
		nd, ch = synth.NewModel(g, d)
	default:
		return fmt.Errorf("unknown op %q", *op)
	}
	if err := nd.WriteFile(*out); err != nil {
		return err
	}
	fmt.Printf("%s: %d upserted, %d deleted -> %s (%d rows)\n", *op, len(ch.Upserted), len(ch.Deleted), *out, nd.N())
	return nil
}

// diffDatasets computes upserts/deletes between two datasets (by id, bitwise).
func diffDatasets(old, cur *vec.Dataset) (*vec.Dataset, []uint64) {
	oi := old.Index()
	ups := vec.New(cur.Dim, 0)
	seen := map[uint64]bool{}
	for i, id := range cur.IDs {
		seen[id] = true
		j, ok := oi[id]
		if ok {
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

func cmdCommit(r *repo.Repo, args []string) error {
	fs := flag.NewFlagSet("commit", flag.ExitOnError)
	msg := fs.String("m", "", "message")
	branch := fs.String("branch", "", "branch (default HEAD)")
	lossy := fs.Bool("lossy", false, "semantic (lossy) mode")
	pqm := fs.Int("pq-m", 0, "PQ sub-spaces (default dim/8)")
	k := fs.Int("k", 256, "coarse centroids")
	retrain := fs.Bool("retrain", false, "retrain the codebook")
	withHNSW := fs.Bool("hnsw", false, "maintain + snapshot an HNSW index")
	hm := fs.Int("M", 16, "HNSW M")
	efc := fs.Int("efc", 100, "HNSW efConstruction")
	pos := parse(fs, args)
	if len(pos) != 1 {
		return errors.New("usage: commit <data.vdv|.fvecs> [flags]")
	}
	t0 := time.Now()
	d, err := vec.ReadFile(pos[0])
	if err != nil {
		return err
	}
	tRead := time.Since(t0)
	opt := repo.CommitOptions{Branch: *branch, Message: *msg, Lossy: repo.LossyConfig{K: *k, M: *pqm, Retrain: *retrain}}
	if *lossy {
		opt.Mode = repo.Lossy
	}
	var g *hnsw.Index
	if *withHNSW {
		b := *branch
		if b == "" {
			b = r.HEAD()
		}
		head, _ := r.BranchHead(b)
		var parentDS *vec.Dataset
		if !head.IsZero() {
			var pg *hnsw.Index
			parentDS, pg, err = r.Checkout(head)
			if err != nil {
				return err
			}
			g = pg
		}
		t1 := time.Now()
		if g == nil {
			g = hnsw.New(d.Dim, hnsw.Config{M: *hm, EfConstruction: *efc, Seed: 42})
			for i, id := range d.IDs {
				g.Insert(id, d.Row(i))
			}
			fmt.Printf("built HNSW over %d vectors in %v\n", d.N(), time.Since(t1).Round(time.Millisecond))
		} else {
			ups, dels := diffDatasets(parentDS, d)
			g.ClearDirty()
			g.Apply(ups, dels)
			fmt.Printf("updated HNSW incrementally (%d upserts, %d deletes, %d nodes touched) in %v\n",
				ups.N(), len(dels), len(g.Dirty()), time.Since(t1).Round(time.Millisecond))
		}
	}
	st, err := r.Commit(d, g, opt)
	if err != nil {
		return err
	}
	fmt.Printf("[%s] snapshot %s  (%d vectors, %s mode)\n", r.HEAD(), st.Snapshot.Short(), d.N(), opt.Mode)
	fmt.Printf("  vector segments: %d total | %d reused | %d delta | %d full\n", st.Segments, st.SegReused, st.SegDelta, st.SegFull)
	if g != nil {
		fmt.Printf("  graph  segments: %d total | %d reused | %d delta | %d full   edges +%d -%d\n",
			st.GSegments, st.GReused, st.GDelta, st.GFull, st.Graph.EdgesAdded, st.Graph.EdgesRemoved)
	}
	fmt.Printf("  new objects: %d (%s)  deduplicated objects: %d\n", st.NewObjects, human(st.NewBytes), st.DedupObjects)
	fmt.Printf("  read %v, commit %v\n", tRead.Round(time.Millisecond), st.Duration.Round(time.Millisecond))
	return nil
}

func cmdLog(r *repo.Repo, args []string) error {
	spec := ""
	if len(args) > 0 {
		spec = args[0]
	}
	hs, err := r.Log(spec, 50)
	if err != nil {
		return err
	}
	for _, h := range hs {
		s, err := r.Snapshot(h)
		if err != nil {
			return err
		}
		g := ""
		if s.HasGraph() {
			g = " +hnsw"
		}
		fmt.Printf("%s  %s  %8d vec  %-8s%s  %s\n", h.Short(), time.Unix(0, s.Time).Format("2006-01-02 15:04:05"), s.Count, s.Mode, g, s.Message)
	}
	return nil
}

func cmdBranch(r *repo.Repo, args []string) error {
	if len(args) == 0 {
		bs, err := r.Branches()
		if err != nil {
			return err
		}
		names := make([]string, 0, len(bs))
		for n := range bs {
			names = append(names, n)
		}
		sort.Strings(names)
		for _, n := range names {
			mark := "  "
			if n == r.HEAD() {
				mark = "* "
			}
			fmt.Printf("%s%-20s %s\n", mark, n, bs[n].Short())
		}
		return nil
	}
	from := ""
	if len(args) > 1 {
		from = args[1]
	}
	h, err := r.Resolve(from)
	if err != nil {
		return err
	}
	t0 := time.Now()
	if err := r.CreateBranch(args[0], h); err != nil {
		return err
	}
	fmt.Printf("created branch %s at %s in %v (zero-copy)\n", args[0], h.Short(), time.Since(t0))
	return nil
}

func cmdCheckout(r *repo.Repo, args []string) error {
	fs := flag.NewFlagSet("checkout", flag.ExitOnError)
	out := fs.String("o", "", "output .vdv")
	pos := parse(fs, args)
	if len(pos) != 1 || *out == "" {
		return errors.New("usage: checkout <ref> -o <out.vdv>")
	}
	h, err := r.Resolve(pos[0])
	if err != nil {
		return err
	}
	t0 := time.Now()
	d, g, err := r.Checkout(h)
	if err != nil {
		return err
	}
	el := time.Since(t0)
	if err := d.WriteFile(*out); err != nil {
		return err
	}
	extra := ""
	if g != nil {
		extra = fmt.Sprintf(" + HNSW index (%d nodes) restored", g.Len())
	}
	fmt.Printf("checked out %s: %d vectors%s in %v -> %s\n", h.Short(), d.N(), extra, el.Round(time.Millisecond), *out)
	return nil
}

func cmdDiff(r *repo.Repo, args []string) error {
	if len(args) != 2 {
		return errors.New("usage: diff <refA> <refB>")
	}
	a, err := r.Resolve(args[0])
	if err != nil {
		return err
	}
	b, err := r.Resolve(args[1])
	if err != nil {
		return err
	}
	d, err := r.Diff(a, b)
	if err != nil {
		return err
	}
	fmt.Printf("%s -> %s\n", a.Short(), b.Short())
	fmt.Printf("  vectors: +%d added  -%d removed  ~%d changed\n", d.VecAdded, d.VecRemoved, d.VecChanged)
	fmt.Printf("  graph:   nodes +%d -%d ~%d   edges ΔE+ %d  ΔE- %d\n", d.Graph.NodesAdded, d.Graph.NodesRemoved, d.Graph.NodesModified, d.Graph.EdgesAdded, d.Graph.EdgesRemoved)
	fmt.Printf("  merkle:  %d trees compared, %d skipped by hash\n", d.TreesCompared, d.TreesSkipped)
	return nil
}

func cmdSearch(r *repo.Repo, args []string) error {
	fs := flag.NewFlagSet("search", flag.ExitOnError)
	qf := fs.String("query", "", "query .vdv/.fvecs")
	k := fs.Int("k", 10, "neighbours")
	ef := fs.Int("ef", 100, "beam width")
	pos := parse(fs, args)
	if len(pos) != 1 || *qf == "" {
		return errors.New("usage: search <ref> --query q.vdv [--k 10]")
	}
	h, err := r.Resolve(pos[0])
	if err != nil {
		return err
	}
	q, err := vec.ReadFile(*qf)
	if err != nil {
		return err
	}
	d, g, err := r.Checkout(h)
	if err != nil {
		return err
	}
	if g == nil {
		return errors.New("snapshot has no HNSW index (commit with --hnsw)")
	}
	t0 := time.Now()
	res := metrics.HNSWSearch(g, q, *k, *ef)
	el := time.Since(t0)
	truth := metrics.BruteForce(d, q, *k)
	for i := 0; i < q.N() && i < 3; i++ {
		fmt.Printf("query %d -> %v\n", i, res[i])
	}
	fmt.Printf("%d queries in %v, recall@%d vs exact = %.4f\n", q.N(), el.Round(time.Microsecond), *k, metrics.Recall(res, truth, *k))
	return nil
}

func cmdVerify(r *repo.Repo, args []string) error {
	var targets []cas.Hash
	if len(args) > 0 {
		h, err := r.Resolve(args[0])
		if err != nil {
			return err
		}
		targets = []cas.Hash{h}
	} else {
		bs, err := r.Branches()
		if err != nil {
			return err
		}
		seen := map[cas.Hash]bool{}
		for _, h := range bs {
			hs, _ := r.Log(h.String(), 1<<30)
			for _, x := range hs {
				if !seen[x] {
					seen[x] = true
					targets = append(targets, x)
				}
			}
		}
	}
	for _, h := range targets {
		if err := r.Verify(h); err != nil {
			return fmt.Errorf("%s: %w", h.Short(), err)
		}
		fmt.Printf("ok  %s\n", h.Short())
	}
	fmt.Printf("%d snapshot(s) verified\n", len(targets))
	return nil
}

func cmdStats(r *repo.Repo) error {
	s, err := r.Stats()
	if err != nil {
		return err
	}
	bs, _ := r.Branches()
	names := []string{}
	for n := range bs {
		names = append(names, n)
	}
	sort.Strings(names)
	fmt.Printf("branches:        %s\n", strings.Join(names, ", "))
	fmt.Printf("snapshots:       %d\n", s.Snapshots)
	fmt.Printf("objects:         %d\n", s.Objects)
	fmt.Printf("logical size:    %s\n", human(s.LogicalBytes))
	fmt.Printf("physical size:   %s  (packs %s + metadata %s)\n", human(s.PhysicalBytes), human(s.PackBytes), human(s.MetaBytes))
	fmt.Printf("dedup ratio:     %.2fx\n", s.DedupRatio())
	return nil
}
