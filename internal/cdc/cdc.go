// Package cdc implements the byte-level baselines VecDedup is compared against:
//
//   - FastCDC (Xia et al., USENIX ATC'16): Gear rolling hash, normalized chunking
//     with MaskS/MaskL, sub-minimum skipping. Used by Restic/Borg-style backup
//     tools and representative of variable-size dedup in enterprise storage.
//   - Fixed-size block dedup (ZFS dedup at a given recordsize, or any block-level
//     CoW snapshot), which is what a filesystem snapshot effectively does.
//
// Both are paired with per-chunk zstd, which is how real systems deploy them.
package cdc

import (
	"crypto/sha256"
	"math/bits"
	"math/rand/v2"

	"vecdedup/internal/codec"
)

var gear [256]uint64

func init() {
	r := rand.New(rand.NewPCG(0x5eed, 0xfa57cdc))
	for i := range gear {
		gear[i] = r.Uint64()
	}
}

// Params configure FastCDC.
type Params struct {
	Min, Avg, Max int
}

// DefaultParams: 2 KiB / 8 KiB / 64 KiB, the paper's recommended setup.
func DefaultParams() Params { return Params{Min: 2 << 10, Avg: 8 << 10, Max: 64 << 10} }

// spreadMask returns a mask with `n` one-bits spread over the high 48 bits, so that
// the bits being judged come from a ~48-byte effective window ("zero padding").
func spreadMask(n int) uint64 {
	var m uint64
	if n <= 0 {
		return 0
	}
	step := 48 / n
	if step < 1 {
		step = 1
	}
	pos := 63
	for i := 0; i < n && pos >= 0; i++ {
		m |= 1 << uint(pos)
		pos -= step
	}
	return m
}

// Chunk splits data into content-defined chunks and returns their lengths.
func Chunk(data []byte, p Params) []int {
	bitsAvg := bits.Len(uint(p.Avg)) - 1
	maskS := spreadMask(bitsAvg + 1) // stricter before the average size
	maskL := spreadMask(bitsAvg - 1) // looser after it (normalization level 1)
	var out []int
	for len(data) > 0 {
		n := cut(data, p, maskS, maskL)
		out = append(out, n)
		data = data[n:]
	}
	return out
}

func cut(src []byte, p Params, maskS, maskL uint64) int {
	n := len(src)
	if n <= p.Min {
		return n
	}
	if n > p.Max {
		n = p.Max
	}
	normal := p.Avg
	if n < normal {
		normal = n
	}
	var fp uint64
	i := p.Min // sub-minimum skipping
	for ; i < normal; i++ {
		fp = (fp << 1) + gear[src[i]]
		if fp&maskS == 0 {
			return i + 1
		}
	}
	for ; i < n; i++ {
		fp = (fp << 1) + gear[src[i]]
		if fp&maskL == 0 {
			return i + 1
		}
	}
	return n
}

// Dedup is a global chunk store simulator: feed it successive snapshot files and it
// accounts logical vs physical (unique, compressed) bytes, exactly like a dedup
// backup repository would.
type Dedup struct {
	seen     map[[32]byte]bool
	Logical  int64
	Physical int64 // unique chunks, zstd-compressed, + 40 B/chunk index overhead
	Chunks   int
	Unique   int
	compress bool
}

// NewDedup creates a store; compress toggles per-chunk zstd.
func NewDedup(compress bool) *Dedup {
	return &Dedup{seen: map[[32]byte]bool{}, compress: compress}
}

const indexOverhead = 40 // hash + location per chunk, typical for dedup indexes

func (d *Dedup) add(chunk []byte) {
	d.Chunks++
	d.Logical += int64(len(chunk))
	h := sha256.Sum256(chunk)
	if d.seen[h] {
		d.Physical += indexOverhead / 2 // reference entry in the snapshot's recipe
		return
	}
	d.seen[h] = true
	d.Unique++
	size := len(chunk)
	if d.compress {
		if c := len(codec.Compress(chunk)); c < size {
			size = c
		}
	}
	d.Physical += int64(size) + indexOverhead
}

// AddCDC ingests one snapshot file using FastCDC.
func (d *Dedup) AddCDC(data []byte, p Params) {
	for _, n := range Chunk(data, p) {
		d.add(data[:n])
		data = data[n:]
	}
}

// AddFixed ingests one snapshot file using fixed-size blocks (ZFS-style dedup).
func (d *Dedup) AddFixed(data []byte, block int) {
	for len(data) > 0 {
		n := min(block, len(data))
		d.add(data[:n])
		data = data[n:]
	}
}

// Ratio = logical / physical.
func (d *Dedup) Ratio() float64 {
	if d.Physical == 0 {
		return 0
	}
	return float64(d.Logical) / float64(d.Physical)
}
