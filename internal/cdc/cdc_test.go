package cdc

import (
	"math/rand/v2"
	"testing"
)

func randBytes(n int, seed uint64) []byte {
	r := rand.New(rand.NewPCG(seed, 2))
	b := make([]byte, n)
	for i := range b {
		b[i] = byte(r.Uint32())
	}
	return b
}

func TestChunkBoundsAndCoverage(t *testing.T) {
	data := randBytes(4<<20, 1)
	p := DefaultParams()
	chunks := Chunk(data, p)
	total := 0
	for i, n := range chunks {
		total += n
		if n > p.Max || (n < p.Min && i != len(chunks)-1) {
			t.Fatalf("chunk %d has size %d outside [%d,%d]", i, n, p.Min, p.Max)
		}
	}
	if total != len(data) {
		t.Fatalf("chunks cover %d of %d bytes", total, len(data))
	}
	avg := float64(total) / float64(len(chunks))
	t.Logf("%d chunks, avg %.0f B", len(chunks), avg)
	if avg < float64(p.Avg)/2 || avg > float64(p.Avg)*2 {
		t.Fatalf("average chunk size %.0f far from target %d", avg, p.Avg)
	}
}

// The defining property of CDC: inserting bytes near the start only disturbs the
// chunks around the edit, so almost everything still dedupes.
func TestCDCResynchronisesAfterInsert(t *testing.T) {
	data := randBytes(2<<20, 3)
	edited := append(append(append([]byte{}, data[:1000]...), []byte("INSERTED")...), data[1000:]...)
	d := NewDedup(false)
	d.AddCDC(data, DefaultParams())
	before := d.Physical
	d.AddCDC(edited, DefaultParams())
	added := d.Physical - before
	if float64(added) > 0.05*float64(len(data)) {
		t.Fatalf("insert cost %d bytes (> 5%% of %d)", added, len(data))
	}
	// fixed blocks, by contrast, are all shifted
	f := NewDedup(false)
	f.AddFixed(data, 8192)
	b2 := f.Physical
	f.AddFixed(edited, 8192)
	if float64(f.Physical-b2) < 0.9*float64(len(data)) {
		t.Fatal("expected fixed-block dedup to lose almost everything after a shift")
	}
}

func TestIdenticalSnapshotsDedupeFully(t *testing.T) {
	data := randBytes(1<<20, 4)
	d := NewDedup(true)
	d.AddCDC(data, DefaultParams())
	d.AddCDC(data, DefaultParams())
	if r := d.Ratio(); r < 1.9 {
		t.Fatalf("two identical snapshots should give ~2x, got %.2f", r)
	}
}
