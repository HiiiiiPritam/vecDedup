package hnsw_test

import (
	"os"
	"testing"

	"vecdedup/internal/hnsw"
	"vecdedup/internal/metrics"
	"vecdedup/internal/vec"
)

// TestSIFTRecall validates the HNSW implementation on real data (SIFT1M subset).
// Skipped unless data/sift/ has been downloaded (see README).
func TestSIFTRecall(t *testing.T) {
	bf, err1 := os.Open("../../data/sift/sift_base_100k.fvecs")
	qf, err2 := os.Open("../../data/sift/sift_query.fvecs")
	if err1 != nil || err2 != nil {
		t.Skip("SIFT data not present")
	}
	defer bf.Close()
	defer qf.Close()
	n := 20000
	if testing.Short() {
		n = 5000
	}
	base, err := vec.ReadFvecs(bf, n)
	if err != nil {
		t.Fatal(err)
	}
	q, err := vec.ReadFvecs(qf, 200)
	if err != nil {
		t.Fatal(err)
	}
	h := hnsw.New(base.Dim, hnsw.Config{M: 16, EfConstruction: 100, Seed: 1})
	for i, id := range base.IDs {
		h.Insert(id, base.Row(i))
	}
	truth := metrics.BruteForce(base, q, 10)
	for _, ef := range []int{16, 32, 64, 128} {
		r := metrics.Recall(metrics.HNSWSearch(h, q, 10, ef), truth, 10)
		t.Logf("SIFT %d: ef=%d recall@10=%.4f", n, ef, r)
		if ef == 128 && r < 0.95 {
			t.Fatalf("recall@10 at ef=128 is %.4f; expected ≥ 0.95 on SIFT", r)
		}
	}
}
