package hnsw_test

import (
	"testing"

	"vecdedup/internal/hnsw"
	"vecdedup/internal/metrics"
	"vecdedup/internal/synth"
	"vecdedup/internal/vec"
)

func build(t *testing.T, n, dim int) (*hnsw.Index, *vec.Dataset, *synth.Generator) {
	t.Helper()
	g := synth.NewGenerator(dim, 20, 0.5, 7)
	d := g.Dataset(n)
	h := hnsw.New(dim, hnsw.Config{M: 12, EfConstruction: 80, Seed: 1})
	for i, id := range d.IDs {
		h.Insert(id, d.Row(i))
	}
	return h, d, g
}

func checkInvariants(t *testing.T, h *hnsw.Index) {
	t.Helper()
	for _, id := range h.IDs() {
		n, _ := h.NodeAdj(id)
		if n.Level != h.LevelFor(id) {
			t.Fatalf("node %d level %d != deterministic level %d", id, n.Level, h.LevelFor(id))
		}
		for l, lst := range n.Layers {
			max := 12
			if l == 0 {
				max = 24
			}
			if len(lst) > max {
				t.Fatalf("node %d layer %d has %d > %d links", id, l, len(lst), max)
			}
			for i, nb := range lst {
				if !h.Contains(nb) {
					t.Fatalf("node %d has dangling edge to %d", id, nb)
				}
				if nb == id {
					t.Fatalf("self loop on %d", id)
				}
				if i > 0 && lst[i-1] >= nb {
					t.Fatalf("neighbour list of %d not strictly sorted", id)
				}
				m, _ := h.NodeAdj(nb)
				if m.Level < l {
					t.Fatalf("edge %d->%d on layer %d but target level is %d", id, nb, l, m.Level)
				}
			}
		}
	}
}

func TestRecallAgainstBruteForce(t *testing.T) {
	h, d, g := build(t, 4000, 32)
	checkInvariants(t, h)
	q := g.Dataset(100)
	truth := metrics.BruteForce(d, q, 10)
	r := metrics.Recall(metrics.HNSWSearch(h, q, 10, 100), truth, 10)
	t.Logf("recall@10 = %.3f", r)
	if r < 0.9 {
		t.Fatalf("recall@10 too low: %.3f", r)
	}
}

func TestDeleteRepairsGraph(t *testing.T) {
	h, d, g := build(t, 3000, 32)
	var dels []uint64
	for i := 0; i < 3000; i += 7 {
		dels = append(dels, d.IDs[i])
	}
	h.DeleteBatch(dels)
	if h.Len() != 3000-len(dels) {
		t.Fatalf("len %d", h.Len())
	}
	checkInvariants(t, h)
	live, _ := synth.Delete(g, d, 0) // copy
	dead := map[uint64]bool{}
	for _, id := range dels {
		dead[id] = true
	}
	kept := vec.New(d.Dim, d.N())
	for i, id := range live.IDs {
		if !dead[id] {
			kept.Append(id, live.Row(i))
		}
	}
	q := g.Dataset(80)
	found := metrics.HNSWSearch(h, q, 10, 100)
	for _, f := range found {
		for _, id := range f {
			if dead[id] {
				t.Fatal("deleted id returned by search")
			}
		}
	}
	r := metrics.Recall(found, metrics.BruteForce(kept, q, 10), 10)
	t.Logf("recall after deleting %d nodes: %.3f", len(dels), r)
	if r < 0.85 {
		t.Fatalf("recall after delete too low: %.3f", r)
	}
}

func TestDeletingEntryPointKeepsIndexUsable(t *testing.T) {
	h, _, g := build(t, 500, 16)
	hd := h.Header()
	h.DeleteBatch([]uint64{hd.EntryID})
	if h.Header().EntryID == hd.EntryID {
		t.Fatal("entry point not replaced")
	}
	if res := h.Search(g.Vector(), 5, 50); len(res) != 5 {
		t.Fatal("search broken after deleting entry point")
	}
}

func TestRestoreGivesIdenticalSearchResults(t *testing.T) {
	h, d, g := build(t, 2000, 24)
	adj := map[uint64]hnsw.Node{}
	for _, id := range h.IDs() {
		adj[id], _ = h.NodeAdj(id)
	}
	h2, err := hnsw.Restore(h.Header(), d, adj)
	if err != nil {
		t.Fatal(err)
	}
	q := g.Dataset(50)
	if !metrics.SameResults(metrics.HNSWSearch(h, q, 10, 64), metrics.HNSWSearch(h2, q, 10, 64)) {
		t.Fatal("restored index returns different results")
	}
}

func TestDirtyTrackingCoversAllChangedNodes(t *testing.T) {
	h, _, g := build(t, 1500, 16)
	before := map[uint64]hnsw.Node{}
	for _, id := range h.IDs() {
		before[id], _ = h.NodeAdj(id)
	}
	h.ClearDirty()
	up := vec.New(16, 10)
	for i := 0; i < 10; i++ {
		up.Append(uint64(5000+i), g.Vector())
	}
	up.Append(3, g.Vector()) // update existing
	h.Apply(up, []uint64{10, 11, 12})
	dirty := map[uint64]bool{}
	for _, id := range h.Dirty() {
		dirty[id] = true
	}
	for _, id := range h.IDs() {
		n, _ := h.NodeAdj(id)
		old, existed := before[id]
		changed := !existed || old.Level != n.Level || len(old.Layers) != len(n.Layers)
		for l := 0; !changed && l < len(n.Layers); l++ {
			if len(old.Layers[l]) != len(n.Layers[l]) {
				changed = true
				break
			}
			for i := range n.Layers[l] {
				if old.Layers[l][i] != n.Layers[l][i] {
					changed = true
				}
			}
		}
		if changed && !dirty[id] {
			t.Fatalf("node %d changed but is not dirty", id)
		}
	}
	for _, id := range []uint64{10, 11, 12} {
		if !dirty[id] {
			t.Fatalf("deleted node %d not dirty", id)
		}
	}
	t.Logf("%d dirty of %d nodes after 14 changes", len(dirty), h.Len())
}
