package codec

import (
	"math"
	"math/rand/v2"
	"testing"
)

func randFloats(n int, seed uint64) []float32 {
	r := rand.New(rand.NewPCG(seed, 1))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64())
	}
	return out
}

func TestShuffleRoundTripIncludingSpecialValues(t *testing.T) {
	fs := randFloats(1000, 1)
	fs = append(fs, float32(math.Inf(1)), float32(math.Inf(-1)), float32(math.NaN()), 0, float32(math.Copysign(0, -1)), math.SmallestNonzeroFloat32)
	out := make([]float32, len(fs))
	if err := UnshuffleFloats(ShuffleFloats(fs), out); err != nil {
		t.Fatal(err)
	}
	for i := range fs {
		if math.Float32bits(fs[i]) != math.Float32bits(out[i]) {
			t.Fatalf("index %d: %x != %x", i, math.Float32bits(fs[i]), math.Float32bits(out[i]))
		}
	}
}

func TestXorIsBitExactInverse(t *testing.T) {
	a, b := randFloats(512, 2), randFloats(512, 3)
	x, back := make([]float32, 512), make([]float32, 512)
	XorFloats(x, a, b)
	XorFloats(back, x, b)
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(back[i]) {
			t.Fatal("xor not invertible")
		}
	}
}

// The core claim of lossless delta mode: XOR deltas of near-identical vectors
// compress far better than the vectors themselves.
func TestXorDeltaOfNoisyCopyCompressesWell(t *testing.T) {
	a := randFloats(64*1024, 4)
	b := make([]float32, len(a))
	r := rand.New(rand.NewPCG(5, 5))
	for i := range a {
		b[i] = a[i] * float32(1+1e-6*r.NormFloat64())
	}
	x := make([]float32, len(a))
	XorFloats(x, b, a)
	raw := len(Compress(ShuffleFloats(b)))
	delta := len(Compress(ShuffleFloats(x)))
	t.Logf("raw=%d delta=%d (%.1fx better)", raw, delta, float64(raw)/float64(delta))
	if delta*3 > raw {
		t.Fatalf("delta (%d) should be at least 3x smaller than raw (%d)", delta, raw)
	}
}

func TestCompressRoundTrip(t *testing.T) {
	in := ShuffleFloats(randFloats(4096, 6))
	out, err := Decompress(Compress(in))
	if err != nil || string(out) != string(in) {
		t.Fatal("zstd round trip failed")
	}
}

func TestWriterReader(t *testing.T) {
	var w Writer
	w.U8(7)
	w.Uvarint(300)
	w.Varint(-5)
	w.U32(0xdeadbeef)
	w.U64(1 << 60)
	w.Bytes([]byte("hello"))
	w.SortedU64s([]uint64{3, 10, 11, 1 << 40})
	r := Reader{B: w.B}
	if r.U8() != 7 || r.Uvarint() != 300 || r.Varint() != -5 || r.U32() != 0xdeadbeef || r.U64() != 1<<60 || string(r.Bytes()) != "hello" {
		t.Fatal("scalar mismatch")
	}
	xs := r.SortedU64s()
	if len(xs) != 4 || xs[3] != 1<<40 || r.Err() != nil || len(r.B) != 0 {
		t.Fatalf("list mismatch %v %v", xs, r.Err())
	}
}

func TestReaderRejectsTruncatedAndHugeLengths(t *testing.T) {
	var w Writer
	w.Uvarint(1 << 40) // claims a giant list
	r := Reader{B: w.B}
	if xs := r.SortedU64s(); len(xs) != 0 || r.Err() == nil {
		t.Fatal("expected corrupt error for huge length")
	}
	r2 := Reader{B: []byte{1, 2}}
	r2.U64()
	if r2.Err() == nil {
		t.Fatal("expected corrupt error for truncated input")
	}
}
