// Package codec holds the low-level, lossless encoding primitives:
//
//   - byte-plane shuffling of float32 streams (groups sign/exponent bytes together)
//   - XOR deltas between two versions of the same vector
//   - zstd block compression
//   - varint / zig-zag helpers and a tiny binary writer/reader
package codec

import (
	"encoding/binary"
	"errors"
	"math"
	"sync"

	"github.com/klauspost/compress/zstd"
)

// ---------------------------------------------------------------------------
// Byte-plane shuffle
// ---------------------------------------------------------------------------

// ShuffleFloats writes the little-endian bytes of fs as 4 planes:
// plane k holds byte k of every float. Byte 3 (sign + high exponent) and byte 2
// (low exponent + high mantissa) are highly repetitive for embeddings, and for XOR
// deltas planes 1–3 are almost entirely zero, which zstd then crushes.
func ShuffleFloats(fs []float32) []byte {
	n := len(fs)
	out := make([]byte, 4*n)
	for i, f := range fs {
		u := math.Float32bits(f)
		out[i] = byte(u)
		out[n+i] = byte(u >> 8)
		out[2*n+i] = byte(u >> 16)
		out[3*n+i] = byte(u >> 24)
	}
	return out
}

// UnshuffleFloats inverts ShuffleFloats into dst (len(dst)*4 == len(b)).
func UnshuffleFloats(b []byte, dst []float32) error {
	n := len(dst)
	if len(b) != 4*n {
		return errors.New("codec: shuffle size mismatch")
	}
	for i := range dst {
		u := uint32(b[i]) | uint32(b[n+i])<<8 | uint32(b[2*n+i])<<16 | uint32(b[3*n+i])<<24
		dst[i] = math.Float32frombits(u)
	}
	return nil
}

// XorFloats sets dst[i] = bits(a[i]) XOR bits(b[i]) reinterpreted as float32.
// XOR is its own inverse: XorFloats(x, XorFloats(a,b), b) == a, bit-exactly,
// which is what makes the lossless delta mode truly lossless (unlike a - b).
func XorFloats(dst, a, b []float32) {
	for i := range dst {
		dst[i] = math.Float32frombits(math.Float32bits(a[i]) ^ math.Float32bits(b[i]))
	}
}

// ---------------------------------------------------------------------------
// zstd
// ---------------------------------------------------------------------------

var (
	encOnce sync.Once
	enc     *zstd.Encoder
	dec     *zstd.Decoder
)

func initZstd() {
	var err error
	enc, err = zstd.NewWriter(nil, zstd.WithEncoderLevel(zstd.SpeedDefault), zstd.WithEncoderConcurrency(1))
	if err != nil {
		panic(err)
	}
	dec, err = zstd.NewReader(nil, zstd.WithDecoderConcurrency(0))
	if err != nil {
		panic(err)
	}
}

// Compress zstd-compresses b. Safe for concurrent use.
func Compress(b []byte) []byte {
	encOnce.Do(initZstd)
	return enc.EncodeAll(b, make([]byte, 0, len(b)/2+64))
}

// Decompress inverts Compress.
func Decompress(b []byte) ([]byte, error) {
	encOnce.Do(initZstd)
	return dec.DecodeAll(b, nil)
}

// ---------------------------------------------------------------------------
// Binary writer / reader with varints
// ---------------------------------------------------------------------------

// Writer is an append-only byte buffer with typed helpers.
type Writer struct{ B []byte }

func (w *Writer) U8(v byte)      { w.B = append(w.B, v) }
func (w *Writer) Uvarint(v uint64) { w.B = binary.AppendUvarint(w.B, v) }
func (w *Writer) Varint(v int64) { w.B = binary.AppendVarint(w.B, v) }
func (w *Writer) U32(v uint32)   { w.B = binary.LittleEndian.AppendUint32(w.B, v) }
func (w *Writer) U64(v uint64)   { w.B = binary.LittleEndian.AppendUint64(w.B, v) }
func (w *Writer) Raw(b []byte)   { w.B = append(w.B, b...) }

// Bytes writes a length-prefixed byte slice.
func (w *Writer) Bytes(b []byte) { w.Uvarint(uint64(len(b))); w.B = append(w.B, b...) }

// SortedU64s writes a strictly increasing list as delta varints (compact for dense ids).
func (w *Writer) SortedU64s(xs []uint64) {
	w.Uvarint(uint64(len(xs)))
	var prev uint64
	for i, x := range xs {
		if i == 0 {
			w.Uvarint(x)
		} else {
			w.Uvarint(x - prev)
		}
		prev = x
	}
}

// ErrCorrupt is returned for malformed encodings.
var ErrCorrupt = errors.New("codec: corrupt data")

// Reader decodes what Writer wrote. The first error sticks; check Err() at the end.
type Reader struct {
	B   []byte
	err error
}

func (r *Reader) Err() error { return r.err }
func (r *Reader) fail() {
	if r.err == nil {
		r.err = ErrCorrupt
	}
	r.B = nil
}

func (r *Reader) U8() byte {
	if len(r.B) < 1 {
		r.fail()
		return 0
	}
	v := r.B[0]
	r.B = r.B[1:]
	return v
}

func (r *Reader) Uvarint() uint64 {
	v, n := binary.Uvarint(r.B)
	if n <= 0 {
		r.fail()
		return 0
	}
	r.B = r.B[n:]
	return v
}

func (r *Reader) Varint() int64 {
	v, n := binary.Varint(r.B)
	if n <= 0 {
		r.fail()
		return 0
	}
	r.B = r.B[n:]
	return v
}

func (r *Reader) U32() uint32 {
	if len(r.B) < 4 {
		r.fail()
		return 0
	}
	v := binary.LittleEndian.Uint32(r.B)
	r.B = r.B[4:]
	return v
}

func (r *Reader) U64() uint64 {
	if len(r.B) < 8 {
		r.fail()
		return 0
	}
	v := binary.LittleEndian.Uint64(r.B)
	r.B = r.B[8:]
	return v
}

func (r *Reader) Raw(n int) []byte {
	if n < 0 || len(r.B) < n {
		r.fail()
		return nil
	}
	v := r.B[:n]
	r.B = r.B[n:]
	return v
}

func (r *Reader) Bytes() []byte {
	n := r.Uvarint()
	if n > uint64(len(r.B)) {
		r.fail()
		return nil
	}
	return r.Raw(int(n))
}

// Len reads a count and sanity-checks it against the remaining input (each element
// needs at least minElemBytes), preventing huge allocations from corrupt data.
func (r *Reader) Len(minElemBytes int) int {
	n := r.Uvarint()
	if minElemBytes < 1 {
		minElemBytes = 1
	}
	if n > uint64(len(r.B)/minElemBytes)+1 {
		r.fail()
		return 0
	}
	return int(n)
}

func (r *Reader) SortedU64s() []uint64 {
	n := r.Len(1)
	xs := make([]uint64, n)
	var prev uint64
	for i := range xs {
		d := r.Uvarint()
		if i == 0 {
			xs[i] = d
		} else {
			xs[i] = prev + d
		}
		prev = xs[i]
	}
	return xs
}
