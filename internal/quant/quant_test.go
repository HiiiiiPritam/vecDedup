package quant_test

import (
	"testing"

	"vecdedup/internal/quant"
	"vecdedup/internal/synth"
	"vecdedup/internal/vec"
)

func mse(cb *quant.Codebook, d *vec.Dataset) float64 {
	code := make([]byte, cb.CodeSize())
	rec := make([]float32, d.Dim)
	var s float64
	for i := 0; i < d.N(); i++ {
		cb.Encode(d.Row(i), code)
		cb.Decode(code, rec)
		s += float64(vec.L2Sq(rec, d.Row(i)))
	}
	return s / float64(d.N())
}

func TestMoreSubspacesLowerError(t *testing.T) {
	g := synth.NewGenerator(64, 16, 0.5, 3)
	d := g.Dataset(3000)
	prev := 1e9
	for _, m := range []int{4, 8, 16} {
		cb, err := quant.Train(d, 32, m, 1)
		if err != nil {
			t.Fatal(err)
		}
		e := mse(cb, d)
		t.Logf("M=%d code=%dB mse=%.5f", m, cb.CodeSize(), e)
		if e >= prev {
			t.Fatalf("error did not decrease with M=%d", m)
		}
		prev = e
	}
	if prev > 0.1 {
		t.Fatalf("reconstruction error too high: %f", prev)
	}
}

func TestCodebookMarshalRoundTrip(t *testing.T) {
	g := synth.NewGenerator(32, 8, 0.5, 4)
	d := g.Dataset(1000)
	cb, err := quant.Train(d, 16, 8, 2)
	if err != nil {
		t.Fatal(err)
	}
	cb2, err := quant.UnmarshalCodebook(cb.Marshal())
	if err != nil {
		t.Fatal(err)
	}
	a, b := make([]byte, cb.CodeSize()), make([]byte, cb.CodeSize())
	for i := 0; i < 100; i++ {
		cb.Encode(d.Row(i), a)
		cb2.Encode(d.Row(i), b)
		if string(a) != string(b) {
			t.Fatal("codebooks disagree after round trip")
		}
	}
	if _, err := quant.UnmarshalCodebook(cb.Marshal()[:20]); err == nil {
		t.Fatal("truncated codebook accepted")
	}
}

func TestTrainRejectsBadParams(t *testing.T) {
	d := synth.NewGenerator(30, 4, 0.5, 5).Dataset(500)
	if _, err := quant.Train(d, 16, 7, 1); err == nil {
		t.Fatal("expected error for Dim % M != 0")
	}
	if _, err := quant.Train(synth.NewGenerator(32, 4, 0.5, 5).Dataset(100), 16, 8, 1); err == nil {
		t.Fatal("expected error for too few training vectors")
	}
}
