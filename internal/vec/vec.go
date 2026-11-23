// Package vec defines the in-memory vector dataset and its on-disk formats.
//
// A Dataset is a set of (ID, vector) rows with a fixed dimension. Rows are stored
// row-major in a single []float32 for cache-friendly access.
package vec

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
)

// Dataset is a collection of vectors keyed by a 64-bit ID.
type Dataset struct {
	Dim  int
	IDs  []uint64
	Data []float32 // len(Data) == len(IDs)*Dim
}

// New allocates an empty dataset with capacity for n rows.
func New(dim, n int) *Dataset {
	return &Dataset{Dim: dim, IDs: make([]uint64, 0, n), Data: make([]float32, 0, n*dim)}
}

// N returns the number of rows.
func (d *Dataset) N() int { return len(d.IDs) }

// Row returns the i-th vector (aliases the underlying storage).
func (d *Dataset) Row(i int) []float32 { return d.Data[i*d.Dim : (i+1)*d.Dim] }

// Append adds a row (the vector is copied).
func (d *Dataset) Append(id uint64, v []float32) {
	if len(v) != d.Dim {
		panic(fmt.Sprintf("vec: dim mismatch %d != %d", len(v), d.Dim))
	}
	d.IDs = append(d.IDs, id)
	d.Data = append(d.Data, v...)
}

// Clone returns a deep copy.
func (d *Dataset) Clone() *Dataset {
	c := &Dataset{Dim: d.Dim, IDs: make([]uint64, len(d.IDs)), Data: make([]float32, len(d.Data))}
	copy(c.IDs, d.IDs)
	copy(c.Data, d.Data)
	return c
}

// SizeBytes is the logical (uncompressed) size: 8 bytes per ID + 4 per float.
func (d *Dataset) SizeBytes() int64 { return int64(len(d.IDs))*8 + int64(len(d.Data))*4 }

// SortByID returns a copy whose rows are sorted by ID (canonical order).
func (d *Dataset) SortByID() *Dataset {
	idx := make([]int, d.N())
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(a, b int) bool { return d.IDs[idx[a]] < d.IDs[idx[b]] })
	out := New(d.Dim, d.N())
	for _, i := range idx {
		out.Append(d.IDs[i], d.Row(i))
	}
	return out
}

// Index builds an ID -> row map.
func (d *Dataset) Index() map[uint64]int {
	m := make(map[uint64]int, d.N())
	for i, id := range d.IDs {
		m[id] = i
	}
	return m
}

// BitEqual reports whether two datasets hold the same ID->vector mapping, bit for bit
// (row order is ignored).
func BitEqual(a, b *Dataset) error {
	if a.Dim != b.Dim {
		return fmt.Errorf("dim %d != %d", a.Dim, b.Dim)
	}
	if a.N() != b.N() {
		return fmt.Errorf("rows %d != %d", a.N(), b.N())
	}
	bi := b.Index()
	for i, id := range a.IDs {
		j, ok := bi[id]
		if !ok {
			return fmt.Errorf("id %d missing", id)
		}
		ra, rb := a.Row(i), b.Row(j)
		for k := range ra {
			if math.Float32bits(ra[k]) != math.Float32bits(rb[k]) {
				return fmt.Errorf("id %d dim %d: %08x != %08x", id, k, math.Float32bits(ra[k]), math.Float32bits(rb[k]))
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// .vdv format: "VDV1" | dim u32 | n u64 | ids [n]u64 | data [n*dim]f32   (little endian)
// This is also the "raw dump" that the FastCDC / ZFS-style baselines snapshot.
// ---------------------------------------------------------------------------

var vdvMagic = [4]byte{'V', 'D', 'V', '1'}

// MarshalVDV serialises the dataset in row order: per row, id then vector. Interleaving
// IDs with vectors mirrors how a vector DB segment/export typically looks.
func (d *Dataset) MarshalVDV() []byte {
	buf := make([]byte, 16+d.N()*(8+4*d.Dim))
	copy(buf, vdvMagic[:])
	binary.LittleEndian.PutUint32(buf[4:], uint32(d.Dim))
	binary.LittleEndian.PutUint64(buf[8:], uint64(d.N()))
	off := 16
	for i, id := range d.IDs {
		binary.LittleEndian.PutUint64(buf[off:], id)
		off += 8
		for _, f := range d.Row(i) {
			binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(f))
			off += 4
		}
	}
	return buf
}

// UnmarshalVDV parses MarshalVDV output.
func UnmarshalVDV(b []byte) (*Dataset, error) {
	if len(b) < 16 || [4]byte(b[:4]) != vdvMagic {
		return nil, errors.New("vec: not a VDV file")
	}
	dim := int(binary.LittleEndian.Uint32(b[4:]))
	n := int(binary.LittleEndian.Uint64(b[8:]))
	if dim <= 0 || len(b) != 16+n*(8+4*dim) {
		return nil, errors.New("vec: corrupt VDV file")
	}
	d := &Dataset{Dim: dim, IDs: make([]uint64, n), Data: make([]float32, n*dim)}
	off := 16
	for i := 0; i < n; i++ {
		d.IDs[i] = binary.LittleEndian.Uint64(b[off:])
		off += 8
		row := d.Data[i*dim : (i+1)*dim]
		for k := range row {
			row[k] = math.Float32frombits(binary.LittleEndian.Uint32(b[off:]))
			off += 4
		}
	}
	return d, nil
}

// WriteFile writes a .vdv file.
func (d *Dataset) WriteFile(path string) error { return os.WriteFile(path, d.MarshalVDV(), 0o644) }

// ReadFile reads a .vdv or .fvecs file (by magic sniffing / extension).
func ReadFile(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var magic [4]byte
	if _, err := io.ReadFull(f, magic[:]); err != nil {
		return nil, err
	}
	if magic == vdvMagic {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		return UnmarshalVDV(b)
	}
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	return ReadFvecs(f, -1)
}

// ReadFvecs parses the TEXMEX .fvecs format (per row: dim int32, then dim float32).
// IDs are assigned 0..n-1. limit < 0 reads everything.
func ReadFvecs(r io.Reader, limit int) (*Dataset, error) {
	br := bufio.NewReaderSize(r, 1<<20)
	var d *Dataset
	var hdr [4]byte
	for i := 0; limit < 0 || i < limit; i++ {
		if _, err := io.ReadFull(br, hdr[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
		dim := int(binary.LittleEndian.Uint32(hdr[:]))
		if d == nil {
			if dim <= 0 || dim > 1<<16 {
				return nil, fmt.Errorf("vec: bad fvecs dim %d", dim)
			}
			d = New(dim, 1024)
		} else if dim != d.Dim {
			return nil, fmt.Errorf("vec: inconsistent fvecs dim %d != %d", dim, d.Dim)
		}
		raw := make([]byte, 4*dim)
		if _, err := io.ReadFull(br, raw); err != nil {
			return nil, err
		}
		row := make([]float32, dim)
		for k := range row {
			row[k] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*k:]))
		}
		d.Append(uint64(i), row)
	}
	if d == nil {
		return nil, errors.New("vec: empty fvecs")
	}
	return d, nil
}

// ---------------------------------------------------------------------------
// Distance kernels
// ---------------------------------------------------------------------------

// L2Sq returns the squared Euclidean distance.
func L2Sq(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		s0 += d0 * d0
		s1 += d1 * d1
		s2 += d2 * d2
		s3 += d3 * d3
	}
	for ; i < n; i++ {
		d := a[i] - b[i]
		s0 += d * d
	}
	return s0 + s1 + s2 + s3
}

// Dot returns the inner product.
func Dot(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a)
	b = b[:n]
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

// Normalize scales v to unit L2 norm in place.
func Normalize(v []float32) {
	n := math.Sqrt(float64(Dot(v, v)))
	if n == 0 {
		return
	}
	inv := float32(1 / n)
	for i := range v {
		v[i] *= inv
	}
}
