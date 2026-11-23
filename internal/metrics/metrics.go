// Package metrics provides ground-truth nearest-neighbour search and Recall@k.
package metrics

import (
	"runtime"
	"sort"
	"sync"

	"vecdedup/internal/hnsw"
	"vecdedup/internal/vec"
)

// BruteForce returns, for each query, the ids of its k exact nearest neighbours
// (L2²) in d. Parallel over queries.
func BruteForce(d *vec.Dataset, queries *vec.Dataset, k int) [][]uint64 {
	out := make([][]uint64, queries.N())
	parallel(queries.N(), func(q int) {
		type hit struct {
			d  float32
			id uint64
		}
		qv := queries.Row(q)
		best := make([]hit, 0, k+1)
		for i := 0; i < d.N(); i++ {
			dist := vec.L2Sq(qv, d.Row(i))
			if len(best) == k && dist >= best[k-1].d {
				continue
			}
			pos := sort.Search(len(best), func(j int) bool { return best[j].d > dist })
			best = append(best, hit{})
			copy(best[pos+1:], best[pos:])
			best[pos] = hit{dist, d.IDs[i]}
			if len(best) > k {
				best = best[:k]
			}
		}
		ids := make([]uint64, len(best))
		for i, h := range best {
			ids[i] = h.id
		}
		out[q] = ids
	})
	return out
}

// HNSWSearch runs an index over all queries.
func HNSWSearch(h *hnsw.Index, queries *vec.Dataset, k, ef int) [][]uint64 {
	out := make([][]uint64, queries.N())
	// the index is read-only during search, so parallel queries are safe
	parallel(queries.N(), func(q int) {
		res := h.Search(queries.Row(q), k, ef)
		ids := make([]uint64, len(res))
		for i, r := range res {
			ids[i] = r.ID
		}
		out[q] = ids
	})
	return out
}

// Recall = mean over queries of |found ∩ truth| / |truth| (truth truncated to k).
func Recall(found, truth [][]uint64, k int) float64 {
	if len(truth) == 0 {
		return 0
	}
	var sum float64
	for q := range truth {
		t := truth[q]
		if len(t) > k {
			t = t[:k]
		}
		set := map[uint64]bool{}
		for _, id := range t {
			set[id] = true
		}
		hit := 0
		f := found[q]
		if len(f) > k {
			f = f[:k]
		}
		for _, id := range f {
			if set[id] {
				hit++
			}
		}
		if len(t) > 0 {
			sum += float64(hit) / float64(len(t))
		}
	}
	return sum / float64(len(truth))
}

// SameResults reports whether two result lists are identical (ids and order).
func SameResults(a, b [][]uint64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if len(a[i]) != len(b[i]) {
			return false
		}
		for j := range a[i] {
			if a[i][j] != b[i][j] {
				return false
			}
		}
	}
	return true
}

func parallel(n int, f func(i int)) {
	w := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	ch := make(chan int, w)
	for t := 0; t < w; t++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range ch {
				f(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		ch <- i
	}
	close(ch)
	wg.Wait()
}
