package segment

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"vecdedup/internal/cas"
)

func randSeg(r *rand.Rand, dim int, ids []uint64) *VecSeg {
	s := &VecSeg{Dim: dim, IDs: ids}
	for range ids {
		for k := 0; k < dim; k++ {
			s.Data = append(s.Data, float32(r.NormFloat64()))
		}
	}
	return s
}

func resolverOf(objs map[cas.Hash]Encoded) Resolver {
	var res Resolver
	res = func(h cas.Hash) (interface{}, error) {
		e, ok := objs[h]
		if !ok {
			return nil, errors.New("missing")
		}
		switch e.Kind {
		case KindVFull, KindVDelta:
			return DecodeVec(e.Kind, e.Payload, res)
		case KindQFull, KindQDelta:
			return DecodeCode(e.Kind, e.Payload, res)
		default:
			return DecodeGraph(e.Kind, e.Payload, res)
		}
	}
	return res
}

func vecEqual(t *testing.T, a, b *VecSeg) {
	t.Helper()
	if len(a.IDs) != len(b.IDs) {
		t.Fatalf("len %d != %d", len(a.IDs), len(b.IDs))
	}
	for i := range a.IDs {
		if a.IDs[i] != b.IDs[i] {
			t.Fatalf("id mismatch at %d", i)
		}
	}
	for i := range a.Data {
		if math.Float32bits(a.Data[i]) != math.Float32bits(b.Data[i]) {
			t.Fatalf("data mismatch at %d", i)
		}
	}
}

func TestVecDeltaChainRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 1))
	dim := 16
	var ids []uint64
	for i := uint64(0); i < 300; i++ {
		ids = append(ids, i*3)
	}
	cur := randSeg(r, dim, ids)
	objs := map[cas.Hash]Encoded{}
	e := EncodeVec(cur, nil, cas.Zero, 0)
	objs[e.Hash] = e
	prevHash, prevDepth := e.Hash, 0
	for step := 0; step < MaxChain+3; step++ {
		next := &VecSeg{Dim: dim}
		for i, id := range cur.IDs {
			switch {
			case r.IntN(20) == 0: // delete
			case r.IntN(10) == 0: // tiny noise (re-embed)
				next.IDs = append(next.IDs, id)
				for _, f := range cur.Row(i) {
					next.Data = append(next.Data, f*(1+1e-6))
				}
			default:
				next.IDs = append(next.IDs, id)
				next.Data = append(next.Data, cur.Row(i)...)
			}
		}
		// insert some ids in the gaps and at the end
		extra := randSeg(r, dim, []uint64{uint64(step*3 + 1), 10000 + uint64(step)})
		merged := &VecSeg{Dim: dim}
		ai := 0
		for i, id := range next.IDs {
			for ai < len(extra.IDs) && extra.IDs[ai] < id {
				merged.IDs = append(merged.IDs, extra.IDs[ai])
				merged.Data = append(merged.Data, extra.Row(ai)...)
				ai++
			}
			merged.IDs = append(merged.IDs, id)
			merged.Data = append(merged.Data, next.Row(i)...)
		}
		for ; ai < len(extra.IDs); ai++ {
			merged.IDs = append(merged.IDs, extra.IDs[ai])
			merged.Data = append(merged.Data, extra.Row(ai)...)
		}
		e := EncodeVec(merged, cur, prevHash, prevDepth)
		if e.Depth > MaxChain {
			t.Fatalf("chain depth %d exceeds max", e.Depth)
		}
		objs[e.Hash] = e
		got, err := resolverOf(objs)(e.Hash)
		if err != nil {
			t.Fatal(err)
		}
		vecEqual(t, merged, got.(*VecSeg))
		if got.(*VecSeg).Hash() != e.Hash {
			t.Fatal("decoded content hash differs from address")
		}
		cur, prevHash, prevDepth = merged, e.Hash, e.Depth
	}
}

func TestVecDeltaSmallerThanFullForSmallChange(t *testing.T) {
	r := rand.New(rand.NewPCG(2, 2))
	var ids []uint64
	for i := uint64(0); i < 1024; i++ {
		ids = append(ids, i)
	}
	base := randSeg(r, 64, ids)
	cur := &VecSeg{Dim: 64, IDs: ids, Data: append([]float32(nil), base.Data...)}
	for k := 0; k < 64; k++ {
		cur.Data[5*64+k] = float32(r.NormFloat64())
	}
	full := EncodeVec(cur, nil, cas.Zero, 0)
	delta := EncodeVec(cur, base, base.Hash(), 0)
	t.Logf("full=%d delta=%d", len(full.Payload), len(delta.Payload))
	if delta.Kind != KindVDelta || len(delta.Payload)*50 > len(full.Payload) {
		t.Fatal("delta not chosen or not small")
	}
	if full.Hash != delta.Hash {
		t.Fatal("content address must not depend on encoding")
	}
}

func TestCodeDeltaRoundTrip(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 3))
	cb := cas.Sum([]byte("cb"))
	mk := func(ids []uint64) *CodeSeg {
		s := &CodeSeg{Codebook: cb, CodeSize: 6, IDs: ids}
		for range ids {
			s.FPs = append(s.FPs, r.Uint64())
			for k := 0; k < 6; k++ {
				s.Codes = append(s.Codes, byte(r.Uint32()))
			}
		}
		return s
	}
	base := mk([]uint64{1, 2, 3, 5, 8, 13, 21})
	cur := &CodeSeg{Codebook: cb, CodeSize: 6}
	for i, id := range base.IDs {
		if id == 5 {
			continue
		}
		cur.IDs = append(cur.IDs, id)
		cur.FPs = append(cur.FPs, base.FPs[i])
		cur.Codes = append(cur.Codes, base.Code(i)...)
		if id == 8 {
			cur.FPs[len(cur.FPs)-1] = 99
			cur.Codes[len(cur.Codes)-1] ^= 0xff
		}
		if id == 13 {
			cur.IDs = append(cur.IDs, 14)
			cur.FPs = append(cur.FPs, 7)
			cur.Codes = append(cur.Codes, 1, 2, 3, 4, 5, 6)
		}
	}
	objs := map[cas.Hash]Encoded{}
	be := EncodeCode(base, nil, cas.Zero, 0)
	objs[be.Hash] = be
	e := EncodeCode(cur, base, be.Hash, 0)
	objs[e.Hash] = e
	got, err := resolverOf(objs)(e.Hash)
	if err != nil {
		t.Fatal(err)
	}
	if got.(*CodeSeg).Hash() != cur.Hash() {
		t.Fatal("code delta round trip mismatch")
	}
}

func TestGraphDeltaRoundTripAndStats(t *testing.T) {
	base := &GraphSeg{Nodes: []GNode{
		{ID: 1, Level: 0, Layers: [][]uint64{{2, 3, 4}}},
		{ID: 2, Level: 1, Layers: [][]uint64{{1, 3}, {9}}},
		{ID: 3, Level: 0, Layers: [][]uint64{{1, 2}}},
		{ID: 4, Level: 0, Layers: [][]uint64{{1}}},
	}}
	cur := &GraphSeg{Nodes: []GNode{
		{ID: 1, Level: 0, Layers: [][]uint64{{2, 3, 5}}},      // -4 +5
		{ID: 2, Level: 1, Layers: [][]uint64{{1, 3}, {9}}},    // same
		{ID: 3, Level: 2, Layers: [][]uint64{{1}, {7}, {8}}},  // level change (re-insert)
		{ID: 5, Level: 0, Layers: [][]uint64{{1, 2}}},         // added (4 removed)
	}}
	objs := map[cas.Hash]Encoded{}
	be := EncodeGraph(base, nil, cas.Zero, 0)
	objs[be.Hash] = be
	e := EncodeGraph(cur, base, be.Hash, 0)
	objs[e.Hash] = e
	got, err := resolverOf(objs)(e.Hash)
	if err != nil {
		t.Fatal(err)
	}
	if got.(*GraphSeg).Hash() != cur.Hash() {
		t.Fatalf("graph delta round trip mismatch: %+v", got)
	}
	d := DiffGraph(base, cur)
	if d.NodesAdded != 1 || d.NodesRemoved != 1 || d.NodesModified != 2 {
		t.Fatalf("unexpected node stats %+v", d)
	}
	// node1: +5 -4; node3: layer0 -2, +layer1 7, +layer2 8; node5: +2 edges; node4: -1 edge
	if d.EdgesAdded != 5 || d.EdgesRemoved != 3 {
		t.Fatalf("unexpected edge stats %+v", d)
	}
}

func TestDecodeRejectsCorruptPayload(t *testing.T) {
	s := randSeg(rand.New(rand.NewPCG(4, 4)), 8, []uint64{1, 2, 3})
	e := EncodeVec(s, nil, cas.Zero, 0)
	bad := append([]byte(nil), e.Payload...)
	bad = bad[:len(bad)-5]
	if _, err := DecodeVec(e.Kind, bad, nil); err == nil {
		t.Fatal("truncated payload accepted")
	}
}
