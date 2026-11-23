// Package quant implements the lossy "semantic mode" codec:
//
//	v  ≈  c[coarse(v)]  +  PQ(v - c[coarse(v)])
//
// i.e. a coarse k-means quantizer (the spec's "shared centroids") followed by a
// Product Quantizer on the residual (Jégou et al., TPAMI 2011 — IVFADC). Each vector
// is stored as a uint16 centroid id plus M one-byte sub-codes.
package quant

import (
	"errors"
	"math"
	"math/rand/v2"
	"runtime"
	"sync"

	"vecdedup/internal/codec"
	"vecdedup/internal/vec"
)

// ---------------------------------------------------------------------------
// k-means (Lloyd with k-means++ seeding)
// ---------------------------------------------------------------------------

// KMeans clusters n points of dimension dim stored row-major in x into k centroids.
// If spherical is true, centroids are re-normalised each iteration (spherical
// k-means; appropriate for cosine-similarity embeddings).
func KMeans(x []float32, dim, k, iters int, spherical bool, seed uint64) []float32 {
	n := len(x) / dim
	if k > n {
		k = n
	}
	rng := rand.New(rand.NewPCG(seed, 0xc0ffee))
	row := func(i int) []float32 { return x[i*dim : (i+1)*dim] }
	c := make([]float32, k*dim)

	// k-means++ seeding on a sample (keeps seeding O(sample*k)).
	sample := n
	if sample > 20000 {
		sample = 20000
	}
	perm := rng.Perm(n)[:sample]
	copy(c[:dim], row(perm[0]))
	d2 := make([]float64, sample)
	for i := range d2 {
		d2[i] = math.Inf(1)
	}
	for j := 1; j < k; j++ {
		var total float64
		last := c[(j-1)*dim : j*dim]
		for s, i := range perm {
			if d := float64(vec.L2Sq(row(i), last)); d < d2[s] {
				d2[s] = d
			}
			total += d2[s]
		}
		target := rng.Float64() * total
		pickIdx := perm[sample-1]
		for s, i := range perm {
			target -= d2[s]
			if target <= 0 {
				pickIdx = i
				break
			}
		}
		copy(c[j*dim:(j+1)*dim], row(pickIdx))
	}

	assign := make([]int32, n)
	for it := 0; it < iters; it++ {
		Assign(x, dim, c, assign)
		sums := make([]float64, k*dim)
		counts := make([]int, k)
		for i := 0; i < n; i++ {
			a := int(assign[i])
			counts[a]++
			r := row(i)
			s := sums[a*dim : (a+1)*dim]
			for t := range r {
				s[t] += float64(r[t])
			}
		}
		for j := 0; j < k; j++ {
			cj := c[j*dim : (j+1)*dim]
			if counts[j] == 0 { // re-seed empty cluster from a random point
				copy(cj, row(rng.IntN(n)))
				continue
			}
			for t := range cj {
				cj[t] = float32(sums[j*dim+t] / float64(counts[j]))
			}
			if spherical {
				vec.Normalize(cj)
			}
		}
	}
	return c
}

// Assign writes the nearest centroid index of every point into out (parallel).
func Assign(x []float32, dim int, c []float32, out []int32) {
	n := len(x) / dim
	k := len(c) / dim
	parallelFor(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			r := x[i*dim : (i+1)*dim]
			best, bd := 0, float32(math.MaxFloat32)
			for j := 0; j < k; j++ {
				if d := vec.L2Sq(r, c[j*dim:(j+1)*dim]); d < bd {
					best, bd = j, d
				}
			}
			out[i] = int32(best)
		}
	})
}

func parallelFor(n int, f func(lo, hi int)) {
	w := runtime.GOMAXPROCS(0)
	if n < 1024 || w == 1 {
		f(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func(lo, hi int) { defer wg.Done(); f(lo, hi) }(lo, hi)
	}
	wg.Wait()
}

// ---------------------------------------------------------------------------
// Residual Product Quantizer
// ---------------------------------------------------------------------------

// Codebook is the trained, shareable part of the lossy codec. It is stored once in
// the CAS and referenced by every snapshot segment that uses it.
type Codebook struct {
	Dim     int
	K       int       // number of coarse centroids (≤ 65536)
	M       int       // number of PQ sub-spaces; Dim % M == 0
	Coarse  []float32 // K*Dim
	Sub     []float32 // M * 256 * (Dim/M)
	// TrainMSE is the mean squared reconstruction error on the training sample. A
	// later snapshot whose vectors reconstruct much worse than this has drifted
	// (e.g. a new embedding model) and triggers retraining.
	TrainMSE float64
	// TrainCoarse is the relative coarse-residual energy E‖v−c*‖² / E‖v‖² on the
	// training sample. It is very sensitive to distribution shift (new clusters,
	// rescaling), while the final PQ error alone can hide a shift.
	TrainCoarse float64
}

// Fit measures, over up to `limit` rows of d, the relative reconstruction error
// E‖v−v̂‖²/E‖v‖² and the relative coarse-residual energy E‖v−c*‖²/E‖v‖².
func (cb *Codebook) Fit(d *vec.Dataset, limit int) (relMSE, relCoarse float64) {
	n := d.N()
	if n == 0 {
		return 0, 0
	}
	step := 1
	if limit > 0 && n > limit {
		step = n / limit
	}
	code := make([]byte, cb.CodeSize())
	rec := make([]float32, cb.Dim)
	var errSum, coarseSum, normSum float64
	for i := 0; i < n; i += step {
		v := d.Row(i)
		cb.Encode(v, code)
		cb.Decode(code, rec)
		errSum += float64(vec.L2Sq(rec, v))
		ci := int(code[0]) | int(code[1])<<8
		coarseSum += float64(vec.L2Sq(v, cb.Coarse[ci*cb.Dim:(ci+1)*cb.Dim]))
		normSum += float64(vec.Dot(v, v))
	}
	if normSum == 0 {
		normSum = 1
	}
	return errSum / normSum, coarseSum / normSum
}

// Drifted reports whether d no longer fits the codebook: its relative error or
// relative coarse residual is more than `factor` times what training achieved.
func (cb *Codebook) Drifted(d *vec.Dataset, factor float64) bool {
	if cb.TrainMSE <= 0 {
		return false
	}
	e, c := cb.Fit(d, 2000)
	return e > factor*cb.TrainMSE || c > factor*cb.TrainCoarse
}

// CodeSize is the encoded size of one vector in bytes.
func (cb *Codebook) CodeSize() int { return 2 + cb.M }

// Train fits a codebook on a sample of the dataset.
func Train(d *vec.Dataset, k, m int, seed uint64) (*Codebook, error) {
	if m <= 0 || d.Dim%m != 0 {
		return nil, errors.New("quant: Dim must be divisible by M")
	}
	if k > 65536 {
		return nil, errors.New("quant: K must be ≤ 65536")
	}
	if d.N() < 256 {
		return nil, errors.New("quant: need at least 256 training vectors")
	}
	x := d.Data
	const maxTrain = 50000
	if d.N() > maxTrain { // subsample for training speed
		rng := rand.New(rand.NewPCG(seed, 1))
		x = make([]float32, 0, maxTrain*d.Dim)
		for _, i := range rng.Perm(d.N())[:maxTrain] {
			x = append(x, d.Row(i)...)
		}
	}
	n := len(x) / d.Dim
	cb := &Codebook{Dim: d.Dim, K: k, M: m}
	cb.Coarse = KMeans(x, d.Dim, k, 15, true, seed)
	cb.K = len(cb.Coarse) / d.Dim

	// residuals
	assign := make([]int32, n)
	Assign(x, d.Dim, cb.Coarse, assign)
	res := make([]float32, len(x))
	for i := 0; i < n; i++ {
		c := cb.Coarse[int(assign[i])*d.Dim:]
		for t := 0; t < d.Dim; t++ {
			res[i*d.Dim+t] = x[i*d.Dim+t] - c[t]
		}
	}
	// per-subspace k-means with 256 centroids
	ds := d.Dim / m
	cb.Sub = make([]float32, m*256*ds)
	sub := make([]float32, n*ds)
	for j := 0; j < m; j++ {
		for i := 0; i < n; i++ {
			copy(sub[i*ds:(i+1)*ds], res[i*d.Dim+j*ds:i*d.Dim+(j+1)*ds])
		}
		cent := KMeans(sub, ds, 256, 12, false, seed+uint64(j)+7)
		copy(cb.Sub[j*256*ds:], cent)
	}
	cb.TrainMSE, cb.TrainCoarse = cb.Fit(&vec.Dataset{Dim: d.Dim, Data: x, IDs: make([]uint64, n)}, 2000)
	return cb, nil
}

// Encode quantizes one vector into dst (len CodeSize()).
func (cb *Codebook) Encode(v []float32, dst []byte) {
	best, bd := 0, float32(math.MaxFloat32)
	for j := 0; j < cb.K; j++ {
		if d := vec.L2Sq(v, cb.Coarse[j*cb.Dim:(j+1)*cb.Dim]); d < bd {
			best, bd = j, d
		}
	}
	dst[0], dst[1] = byte(best), byte(best>>8)
	c := cb.Coarse[best*cb.Dim:]
	ds := cb.Dim / cb.M
	r := make([]float32, ds)
	for j := 0; j < cb.M; j++ {
		for t := 0; t < ds; t++ {
			r[t] = v[j*ds+t] - c[j*ds+t]
		}
		bj, bdj := 0, float32(math.MaxFloat32)
		base := cb.Sub[j*256*ds:]
		for q := 0; q < 256; q++ {
			if d := vec.L2Sq(r, base[q*ds:(q+1)*ds]); d < bdj {
				bj, bdj = q, d
			}
		}
		dst[2+j] = byte(bj)
	}
}

// Decode reconstructs an approximate vector from its code.
func (cb *Codebook) Decode(code []byte, dst []float32) {
	ci := int(code[0]) | int(code[1])<<8
	c := cb.Coarse[ci*cb.Dim : (ci+1)*cb.Dim]
	ds := cb.Dim / cb.M
	for j := 0; j < cb.M; j++ {
		s := cb.Sub[j*256*ds+int(code[2+j])*ds:]
		for t := 0; t < ds; t++ {
			dst[j*ds+t] = c[j*ds+t] + s[t]
		}
	}
}

// EncodeAll encodes many vectors in parallel; out has len n*CodeSize().
func (cb *Codebook) EncodeAll(x []float32) []byte {
	n := len(x) / cb.Dim
	cs := cb.CodeSize()
	out := make([]byte, n*cs)
	parallelFor(n, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			cb.Encode(x[i*cb.Dim:(i+1)*cb.Dim], out[i*cs:(i+1)*cs])
		}
	})
	return out
}

// Marshal serialises the codebook (floats are byte-shuffled for compressibility).
func (cb *Codebook) Marshal() []byte {
	var w codec.Writer
	w.Uvarint(uint64(cb.Dim))
	w.Uvarint(uint64(cb.K))
	w.Uvarint(uint64(cb.M))
	w.Bytes(codec.ShuffleFloats(cb.Coarse))
	w.Bytes(codec.ShuffleFloats(cb.Sub))
	w.U64(math.Float64bits(cb.TrainMSE))
	w.U64(math.Float64bits(cb.TrainCoarse))
	return w.B
}

// UnmarshalCodebook inverts Marshal.
func UnmarshalCodebook(b []byte) (*Codebook, error) {
	r := codec.Reader{B: b}
	cb := &Codebook{Dim: int(r.Uvarint()), K: int(r.Uvarint()), M: int(r.Uvarint())}
	cbytes, sbytes := r.Bytes(), r.Bytes()
	cb.TrainMSE = math.Float64frombits(r.U64())
	cb.TrainCoarse = math.Float64frombits(r.U64())
	if r.Err() != nil || cb.Dim <= 0 || cb.M <= 0 || cb.Dim%cb.M != 0 || len(cbytes) != 4*cb.K*cb.Dim || len(sbytes) != 4*cb.M*256*(cb.Dim/cb.M) {
		return nil, codec.ErrCorrupt
	}
	cb.Coarse = make([]float32, cb.K*cb.Dim)
	cb.Sub = make([]float32, cb.M*256*(cb.Dim/cb.M))
	if err := codec.UnshuffleFloats(cbytes, cb.Coarse); err != nil {
		return nil, err
	}
	if err := codec.UnshuffleFloats(sbytes, cb.Sub); err != nil {
		return nil, err
	}
	return cb, nil
}
