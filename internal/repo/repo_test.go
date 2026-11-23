package repo_test

import (
	"testing"

	"vecdedup/internal/cas"
	"vecdedup/internal/hnsw"
	"vecdedup/internal/metrics"
	"vecdedup/internal/repo"
	"vecdedup/internal/synth"
	"vecdedup/internal/vec"
)

func open(t *testing.T) (*repo.Repo, string) {
	t.Helper()
	dir := t.TempDir()
	r, err := repo.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r, dir
}

func mustCommit(t *testing.T, r *repo.Repo, d *vec.Dataset, g *hnsw.Index, opt repo.CommitOptions) repo.CommitStats {
	t.Helper()
	st, err := r.Commit(d, g, opt)
	if err != nil {
		t.Fatal(err)
	}
	return st
}

// Every snapshot in a churn history must check out bit-exactly.
func TestLosslessHistoryCheckoutIsBitExact(t *testing.T) {
	r, _ := open(t)
	g := synth.NewGenerator(32, 10, 0.5, 1)
	d := g.Dataset(5000)
	var snaps []cas.Hash
	var states []*vec.Dataset
	steps := []func(*vec.Dataset) *vec.Dataset{
		func(x *vec.Dataset) *vec.Dataset { y, _ := synth.Edit(g, x, 0.05); return y },
		func(x *vec.Dataset) *vec.Dataset { y, _ := synth.Append(g, x, 0.1); return y },
		func(x *vec.Dataset) *vec.Dataset { y, _ := synth.Delete(g, x, 0.05); return y },
		func(x *vec.Dataset) *vec.Dataset { y, _ := synth.Reembed(g, x, 1e-6); return y },
		func(x *vec.Dataset) *vec.Dataset { y, _ := synth.Shuffle(g, x); return y },
	}
	st := mustCommit(t, r, d, nil, repo.CommitOptions{Message: "base"})
	snaps, states = append(snaps, st.Snapshot), append(states, d)
	for _, f := range steps {
		d = f(d)
		st = mustCommit(t, r, d, nil, repo.CommitOptions{})
		snaps, states = append(snaps, st.Snapshot), append(states, d)
	}
	// shuffle is not a content change: the snapshot root must reuse every segment
	if st.SegReused != st.Segments {
		t.Fatalf("shuffle re-encoded %d of %d segments", st.Segments-st.SegReused, st.Segments)
	}
	for i, h := range snaps {
		got, _, err := r.Checkout(h)
		if err != nil {
			t.Fatal(err)
		}
		if err := vec.BitEqual(states[i], got); err != nil {
			t.Fatalf("snapshot %d: %v", i, err)
		}
		if err := r.Verify(h); err != nil {
			t.Fatalf("verify %d: %v", i, err)
		}
	}
}

func TestUnchangedCommitIsFree(t *testing.T) {
	r, _ := open(t)
	d := synth.NewGenerator(16, 4, 0.5, 2).Dataset(3000)
	mustCommit(t, r, d, nil, repo.CommitOptions{Time: 1})
	st := mustCommit(t, r, d, nil, repo.CommitOptions{Time: 2})
	if st.SegReused != st.Segments || st.NewObjects != 1 { // only the snapshot object
		t.Fatalf("expected only a new snapshot object, got %+v", st)
	}
}

// The incremental path must produce exactly the same Merkle root as a full re-scan.
func TestIncrementalCommitMatchesFullCommit(t *testing.T) {
	for _, mode := range []repo.Mode{repo.Lossless, repo.Lossy} {
		t.Run(mode.String(), func(t *testing.T) {
			g := synth.NewGenerator(32, 8, 0.5, 3)
			d := g.Dataset(4000)
			d2, ch := synth.Mixed(g, d, 0.1)
			idx := d2.Index()
			ups := vec.New(32, len(ch.Upserted))
			for _, id := range ch.Upserted {
				ups.Append(id, d2.Row(idx[id]))
			}
			opt := repo.CommitOptions{Mode: mode, Time: 7, Lossy: repo.LossyConfig{K: 16, M: 8}}

			r1, _ := open(t)
			mustCommit(t, r1, d, nil, opt)
			full := mustCommit(t, r1, d2, nil, opt)

			r2, _ := open(t)
			mustCommit(t, r2, d, nil, opt)
			inc, err := r2.CommitChanges(ups, ch.Deleted, nil, opt)
			if err != nil {
				t.Fatal(err)
			}
			if full.Snapshot != inc.Snapshot {
				t.Fatalf("full root %s != incremental root %s", full.Snapshot.Short(), inc.Snapshot.Short())
			}
		})
	}
}

func TestGraphSnapshotRestoreAndIncrementalCommit(t *testing.T) {
	r, _ := open(t)
	gen := synth.NewGenerator(24, 10, 0.5, 4)
	d := gen.Dataset(3000)
	h := hnsw.New(24, hnsw.Config{M: 8, EfConstruction: 60, Seed: 9})
	for i, id := range d.IDs {
		h.Insert(id, d.Row(i))
	}
	s0 := mustCommit(t, r, d, h, repo.CommitOptions{})
	q := gen.Dataset(40)
	want0 := metrics.HNSWSearch(h, q, 10, 50)

	d2, ch := synth.Mixed(gen, d, 0.04)
	idx := d2.Index()
	ups := vec.New(24, len(ch.Upserted))
	for _, id := range ch.Upserted {
		ups.Append(id, d2.Row(idx[id]))
	}
	h.Apply(ups, ch.Deleted)
	dirty := len(h.Dirty())
	s1, err := r.CommitChanges(ups, ch.Deleted, h, repo.CommitOptions{})
	if err != nil {
		t.Fatal(err)
	}
	want1 := metrics.HNSWSearch(h, q, 10, 50)
	t.Logf("dirty nodes %d / %d; graph segs delta=%d full=%d reused=%d; edges +%d -%d",
		dirty, h.Len(), s1.GDelta, s1.GFull, s1.GReused, s1.Graph.EdgesAdded, s1.Graph.EdgesRemoved)
	if s1.Graph.EdgesAdded == 0 {
		t.Fatal("expected graph edge changes")
	}

	for _, c := range []struct {
		h    cas.Hash
		want [][]uint64
		ds   *vec.Dataset
	}{{s0.Snapshot, want0, d}, {s1.Snapshot, want1, d2}} {
		gotD, gotG, err := r.Checkout(c.h)
		if err != nil {
			t.Fatal(err)
		}
		if err := vec.BitEqual(c.ds, gotD); err != nil {
			t.Fatal(err)
		}
		if !metrics.SameResults(c.want, metrics.HNSWSearch(gotG, q, 10, 50)) {
			t.Fatal("restored graph answers queries differently")
		}
		if err := r.Verify(c.h); err != nil {
			t.Fatal(err)
		}
	}
	diff, err := r.Diff(s0.Snapshot, s1.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	// ids appended and deleted within the same step never existed in either snapshot
	oldIdx := d.Index()
	realDeletes := 0
	for _, id := range ch.Deleted {
		if _, ok := oldIdx[id]; ok {
			realDeletes++
		}
	}
	if diff.VecAdded+diff.VecRemoved+diff.VecChanged != len(ch.Upserted)+realDeletes || diff.VecRemoved != realDeletes {
		t.Fatalf("diff %+v does not match change (%d upserts, %d deletes)", diff, len(ch.Upserted), realDeletes)
	}
}

func TestBranchesAreIsolatedAndZeroCopy(t *testing.T) {
	r, _ := open(t)
	g := synth.NewGenerator(16, 4, 0.5, 5)
	d := g.Dataset(2000)
	base := mustCommit(t, r, d, nil, repo.CommitOptions{}).Snapshot
	before, _ := r.Stats()
	if err := r.CreateBranch("exp", base); err != nil {
		t.Fatal(err)
	}
	after, _ := r.Stats()
	if after.PackBytes != before.PackBytes {
		t.Fatal("branch creation wrote object data")
	}
	if err := r.CreateBranch("exp", base); err == nil {
		t.Fatal("duplicate branch allowed")
	}
	dExp, _ := synth.Edit(g, d, 0.2)
	expSnap := mustCommit(t, r, dExp, nil, repo.CommitOptions{Branch: "exp"}).Snapshot
	mainHead, _ := r.BranchHead("main")
	if mainHead != base {
		t.Fatal("commit on branch moved main")
	}
	got, _, _ := r.Checkout(mainHead)
	if err := vec.BitEqual(d, got); err != nil {
		t.Fatal(err)
	}
	got, _, _ = r.Checkout(expSnap)
	if err := vec.BitEqual(dExp, got); err != nil {
		t.Fatal(err)
	}
	if h, err := r.Resolve("exp~1"); err != nil || h != base {
		t.Fatal("exp~1 should resolve to the base snapshot")
	}
	if err := r.Switch("nope"); err == nil {
		t.Fatal("switch to missing branch allowed")
	}
}

func TestLossyModeRecallAndCompression(t *testing.T) {
	r, _ := open(t)
	g := synth.NewGenerator(64, 20, 0.5, 6)
	d := g.Dataset(6000)
	st := mustCommit(t, r, d, nil, repo.CommitOptions{Mode: repo.Lossy, Lossy: repo.LossyConfig{K: 64, M: 16}})
	got, _, err := r.Checkout(st.Snapshot)
	if err != nil {
		t.Fatal(err)
	}
	q := g.Dataset(100)
	recall := metrics.Recall(metrics.BruteForce(got, q, 10), metrics.BruteForce(d, q, 10), 10)
	stats, _ := r.Stats()
	t.Logf("lossy recall@10=%.3f, logical=%d physical=%d (%.1fx)", recall, stats.LogicalBytes, stats.PhysicalBytes, stats.DedupRatio())
	if recall < 0.6 {
		t.Fatalf("lossy recall too low: %.3f", recall)
	}
	// next snapshot: 5% edits -> only those vectors re-quantized, codebook reused
	d2, _ := synth.Edit(g, d, 0.05)
	st2 := mustCommit(t, r, d2, nil, repo.CommitOptions{Mode: repo.Lossy})
	if st2.NewBytes*10 > st.NewBytes {
		t.Fatalf("second lossy commit too large: %d vs %d", st2.NewBytes, st.NewBytes)
	}
	if err := r.Verify(st2.Snapshot); err != nil {
		t.Fatal(err)
	}
}

// A new embedding model must not be squeezed through the old codebook.
func TestLossyCodebookDriftTriggersRetrain(t *testing.T) {
	r, _ := open(t)
	g := synth.NewGenerator(32, 10, 0.5, 8)
	d := g.Dataset(3000)
	opt := repo.CommitOptions{Mode: repo.Lossy, Lossy: repo.LossyConfig{K: 32, M: 8}}
	mustCommit(t, r, d, nil, opt)
	d2, _ := synth.Edit(g, d, 0.05)
	if st := mustCommit(t, r, d2, nil, opt); st.Retrained {
		t.Fatal("small edit should reuse the codebook")
	}
	d3, _ := synth.NewModel(g, d2)
	st := mustCommit(t, r, d3, nil, opt)
	if !st.Retrained {
		t.Fatal("new model should trigger retraining")
	}
	got, _, _ := r.Checkout(st.Snapshot)
	q := synth.NewGenerator(32, 10, 0.5, 8).Dataset(50)
	recall := metrics.Recall(metrics.BruteForce(got, q, 10), metrics.BruteForce(d3, q, 10), 10)
	t.Logf("recall after model change with retrain: %.3f", recall)
	if recall < 0.5 {
		t.Fatalf("recall after retrain too low: %.3f", recall)
	}
	// A scale change (e.g. pipeline stops normalising) has LOWER absolute error
	// but must still be detected.
	d4 := d3.Clone()
	for i := range d4.Data {
		d4.Data[i] *= 0.01
	}
	if st := mustCommit(t, r, d4, nil, opt); !st.Retrained {
		t.Fatal("rescaled data should trigger retraining")
	}
}

func TestReopenPersistsEverything(t *testing.T) {
	dir := t.TempDir()
	r, _ := repo.Open(dir)
	d := synth.NewGenerator(16, 4, 0.5, 7).Dataset(1500)
	st, err := r.Commit(d, nil, repo.CommitOptions{Message: "hello"})
	if err != nil {
		t.Fatal(err)
	}
	r.Close()
	r, _ = repo.Open(dir)
	defer r.Close()
	h, err := r.Resolve("main")
	if err != nil || h != st.Snapshot {
		t.Fatal("branch lost after reopen")
	}
	s, _ := r.Snapshot(h)
	if s.Message != "hello" || s.Count != 1500 {
		t.Fatalf("snapshot metadata wrong: %+v", s)
	}
	got, _, _ := r.Checkout(h)
	if err := vec.BitEqual(d, got); err != nil {
		t.Fatal(err)
	}
}

func TestRejectsBadInput(t *testing.T) {
	r, _ := open(t)
	d := vec.New(4, 2)
	d.Append(1, []float32{1, 2, 3, 4})
	d.Append(1, []float32{5, 6, 7, 8})
	if _, err := r.Commit(d, nil, repo.CommitOptions{}); err == nil {
		t.Fatal("duplicate ids accepted")
	}
	ok := vec.New(4, 1)
	ok.Append(1, []float32{1, 2, 3, 4})
	mustCommit(t, r, ok, nil, repo.CommitOptions{})
	other := vec.New(8, 1)
	other.Append(1, make([]float32, 8))
	if _, err := r.Commit(other, nil, repo.CommitOptions{}); err == nil {
		t.Fatal("dimension change accepted")
	}
}
