// Package synth generates deterministic "embedding-like" datasets and simulates the
// churn patterns that real RAG pipelines produce between snapshots.
//
// Embedding-like = a Gaussian mixture projected onto the unit sphere. Real sentence
// embeddings are strongly clustered by topic and L2-normalised, and this mimics both.
package synth

import (
	"math"
	"math/rand/v2"
	"sort"

	"vecdedup/internal/vec"
)

// Generator produces vectors from a fixed set of topic clusters.
//
// Real embeddings have a LOW intrinsic dimension (tens of directions matter even
// in a 768-d space). Each cluster therefore varies mainly along its own random
// low-rank subspace (`Latent` directions), plus a small isotropic noise floor.
// Purely isotropic noise would make every cluster a 768-d ball, which is far
// harder for ANN search than any real embedding set.
type Generator struct {
	Dim     int
	Latent  int
	centers [][]float32
	bases   [][]float32 // per cluster: Latent x Dim
	spread  float64
	rng     *rand.Rand
}

// NewGenerator creates a generator with `clusters` topics. spread controls the
// within-topic variation (0.3–0.6 gives realistic cosine-similarity distributions).
func NewGenerator(dim, clusters int, spread float64, seed uint64) *Generator {
	g := &Generator{Dim: dim, spread: spread, rng: rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))}
	g.Latent = min(16, dim)
	for c := 0; c < clusters; c++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = float32(g.rng.NormFloat64())
		}
		vec.Normalize(v)
		g.centers = append(g.centers, v)
		b := make([]float32, g.Latent*dim)
		for i := range b {
			b[i] = float32(g.rng.NormFloat64() / math.Sqrt(float64(dim)))
		}
		g.bases = append(g.bases, b)
	}
	return g
}

// Vector draws one unit-norm vector.
func (g *Generator) Vector() []float32 {
	ci := g.rng.IntN(len(g.centers))
	c, b := g.centers[ci], g.bases[ci]
	v := make([]float32, g.Dim)
	copy(v, c)
	ls := g.spread / math.Sqrt(float64(g.Latent)) * 1.5
	for k := 0; k < g.Latent; k++ {
		z := float32(g.rng.NormFloat64() * ls)
		row := b[k*g.Dim : (k+1)*g.Dim]
		for i := range v {
			v[i] += z * row[i]
		}
	}
	iso := 0.05 * g.spread / math.Sqrt(float64(g.Dim))
	for i := range v {
		v[i] += float32(g.rng.NormFloat64() * iso)
	}
	vec.Normalize(v)
	return v
}

// Dataset draws n vectors with IDs 0..n-1.
func (g *Generator) Dataset(n int) *vec.Dataset {
	d := vec.New(g.Dim, n)
	for i := 0; i < n; i++ {
		d.Append(uint64(i), g.Vector())
	}
	return d
}

// Rand exposes the generator's RNG (used by churn operations so a whole experiment
// is reproducible from one seed).
func (g *Generator) Rand() *rand.Rand { return g.rng }

// ---------------------------------------------------------------------------
// Churn operations. Each returns a NEW dataset; the input is not modified.
// ---------------------------------------------------------------------------

// Change describes what a churn step did, so callers can apply the same change to
// an HNSW index incrementally.
type Change struct {
	Upserted []uint64 // ids whose vector was added or replaced
	Deleted  []uint64
}

func pick(rng *rand.Rand, n, k int) []int {
	if k > n {
		k = n
	}
	p := rng.Perm(n)[:k]
	sort.Ints(p)
	return p
}

// Edit replaces the vectors of a `frac` fraction of rows with freshly drawn ones
// (a document was edited and re-embedded). IDs and row order are preserved.
func Edit(g *Generator, d *vec.Dataset, frac float64) (*vec.Dataset, Change) {
	out := d.Clone()
	var ch Change
	for _, i := range pick(g.rng, d.N(), int(frac*float64(d.N()))) {
		copy(out.Row(i), g.Vector())
		ch.Upserted = append(ch.Upserted, out.IDs[i])
	}
	return out, ch
}

// Append adds frac*N new rows with new IDs at the end.
func Append(g *Generator, d *vec.Dataset, frac float64) (*vec.Dataset, Change) {
	out := d.Clone()
	var ch Change
	next := uint64(0)
	for _, id := range d.IDs {
		if id >= next {
			next = id + 1
		}
	}
	k := int(frac * float64(d.N()))
	for i := 0; i < k; i++ {
		out.Append(next, g.Vector())
		ch.Upserted = append(ch.Upserted, next)
		next++
	}
	return out, ch
}

// Delete removes frac*N random rows (row order of the survivors is kept, so the
// physical file "compacts": later rows shift left).
func Delete(g *Generator, d *vec.Dataset, frac float64) (*vec.Dataset, Change) {
	del := map[int]bool{}
	var ch Change
	for _, i := range pick(g.rng, d.N(), int(frac*float64(d.N()))) {
		del[i] = true
		ch.Deleted = append(ch.Deleted, d.IDs[i])
	}
	out := vec.New(d.Dim, d.N()-len(del))
	for i := 0; i < d.N(); i++ {
		if !del[i] {
			out.Append(d.IDs[i], d.Row(i))
		}
	}
	return out, ch
}

// Mixed = a realistic day of ingestion: edit + append + delete.
func Mixed(g *Generator, d *vec.Dataset, frac float64) (*vec.Dataset, Change) {
	a, c1 := Edit(g, d, frac*0.5)
	b, c2 := Append(g, a, frac*0.3)
	c, c3 := Delete(g, b, frac*0.2)
	// Deleted ids may have been upserted in this same step; the final state wins.
	dead := map[uint64]bool{}
	for _, id := range c3.Deleted {
		dead[id] = true
	}
	var ch Change
	for _, id := range append(c1.Upserted, c2.Upserted...) {
		if !dead[id] {
			ch.Upserted = append(ch.Upserted, id)
		}
	}
	ch.Deleted = c3.Deleted
	return c, ch
}

// Reembed simulates re-running the embedding pipeline over the WHOLE corpus on
// different hardware / batch sizes: every component gets a relative perturbation
// of about `relNoise` (1e-6 ≈ a few float32 ULPs), then the vector is renormalised.
// Semantically nothing changed, but nearly every float's low mantissa bits did.
// Each vector keeps its original L2 norm (unit vectors stay unit; SIFT-like data
// keeps its scale).
func Reembed(g *Generator, d *vec.Dataset, relNoise float64) (*vec.Dataset, Change) {
	out := d.Clone()
	var ch Change
	for i := 0; i < out.N(); i++ {
		r := out.Row(i)
		norm := float32(math.Sqrt(float64(vec.Dot(r, r))))
		for k := range r {
			r[k] *= float32(1 + relNoise*g.rng.NormFloat64())
		}
		vec.Normalize(r)
		for k := range r {
			r[k] *= norm
		}
		ch.Upserted = append(ch.Upserted, out.IDs[i])
	}
	return out, ch
}

// Shuffle permutes row order without changing any (id, vector) pair. This models
// a vector DB compaction / re-sort, which rewrites the physical file.
func Shuffle(g *Generator, d *vec.Dataset) (*vec.Dataset, Change) {
	out := vec.New(d.Dim, d.N())
	for _, i := range g.rng.Perm(d.N()) {
		out.Append(d.IDs[i], d.Row(i))
	}
	return out, Change{}
}

// NewModel replaces every vector (an embedding-model upgrade). Nobody can dedupe
// this; it is the honest worst case.
func NewModel(g *Generator, d *vec.Dataset) (*vec.Dataset, Change) {
	out := d.Clone()
	g2 := NewGenerator(d.Dim, len(g.centers), g.spread, g.rng.Uint64())
	var ch Change
	for i := 0; i < out.N(); i++ {
		copy(out.Row(i), g2.Vector())
		ch.Upserted = append(ch.Upserted, out.IDs[i])
	}
	return out, ch
}
