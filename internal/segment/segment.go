// Package segment defines the unit of deduplication: a *segment* holds all rows
// whose ID falls in one fixed ID range (id >> Shift). There are three segment types:
//
//	VecSeg   lossless vectors              (kinds KindVFull / KindVDelta)
//	CodeSeg  lossy residual-PQ codes        (kinds KindQFull / KindQDelta)
//	GraphSeg HNSW adjacency of those nodes  (kinds KindGFull / KindGDelta)
//
// Every segment has a CANONICAL byte form whose SHA-256 is its content address. The
// stored payload may be a full encoding or a delta against a *base* segment (the
// same ID range in the parent snapshot). Since the address depends only on the
// logical content, two snapshots that reach identical content via different
// histories still share one object.
package segment

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"

	"github.com/cespare/xxhash/v2"

	"vecdedup/internal/cas"
	"vecdedup/internal/codec"
)

// Shift sets the segment size: 2^8 = 256 consecutive IDs per segment. Smaller
// segments mean random edits touch less data per commit (lower incremental commit
// latency) at the cost of more objects. 256 was chosen from the benchmark sweep.
const Shift = 8

// smallDelta: if a delta is below this fraction of the raw segment size, it is
// accepted without also computing the full encoding (saves CPU on small changes).
const smallDelta = 0.25

// SegOf returns the segment index of an id.
func SegOf(id uint64) uint64 { return id >> Shift }

// MaxChain bounds delta-chain length so reads stay O(MaxChain) decodes.
const MaxChain = 8

// Object kinds stored in the CAS.
const (
	KindVFull byte = 1 + iota
	KindVDelta
	KindQFull
	KindQDelta
	KindGFull
	KindGDelta
	KindCodebook
	KindTree
	KindSnapshot
	KindGraphHeader
)

// KindName is for diagnostics.
func KindName(k byte) string {
	names := []string{"?", "vec-full", "vec-delta", "code-full", "code-delta", "graph-full", "graph-delta", "codebook", "tree", "snapshot", "graph-header"}
	if int(k) < len(names) {
		return names[k]
	}
	return "?"
}

// Resolver loads and decodes a base segment by hash (with caching, chain following).
type Resolver func(h cas.Hash) (interface{}, error)

// Encoded is the result of encoding a segment.
type Encoded struct {
	Hash    cas.Hash
	Kind    byte
	Payload []byte
	Depth   int
}

func header(base cas.Hash, depth int) *codec.Writer {
	w := &codec.Writer{}
	w.Raw(base[:])
	w.Uvarint(uint64(depth))
	return w
}

func readHeader(r *codec.Reader) (cas.Hash, int) {
	var h cas.Hash
	copy(h[:], r.Raw(32))
	return h, int(r.Uvarint())
}

// ===========================================================================
// VecSeg — lossless vectors
// ===========================================================================

// VecSeg is a sorted-by-id set of vectors.
type VecSeg struct {
	Dim  int
	IDs  []uint64
	Data []float32
}

func (s *VecSeg) Row(i int) []float32 { return s.Data[i*s.Dim : (i+1)*s.Dim] }

// Hash returns the content address (sha256 of the canonical form).
func (s *VecSeg) Hash() cas.Hash {
	b := make([]byte, 0, 16+len(s.IDs)*8+len(s.Data)*4)
	b = append(b, 'V')
	b = binary.LittleEndian.AppendUint32(b, uint32(s.Dim))
	b = binary.LittleEndian.AppendUint32(b, uint32(len(s.IDs)))
	for _, id := range s.IDs {
		b = binary.LittleEndian.AppendUint64(b, id)
	}
	for _, f := range s.Data {
		b = binary.LittleEndian.AppendUint32(b, math.Float32bits(f))
	}
	return cas.Sum(b)
}

func rowsEqual(a, b []float32) bool {
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// EncodeVec encodes s, as a delta against base if that is smaller (and the chain is
// short enough). base may be nil.
func EncodeVec(s *VecSeg, base *VecSeg, baseHash cas.Hash, baseDepth int) Encoded {
	h := s.Hash()
	var delta []byte
	if base != nil && base.Dim == s.Dim && baseDepth+1 <= MaxChain {
		delta = encodeVecDelta(s, base, baseHash, baseDepth+1)
		if float64(len(delta)) < smallDelta*float64(len(s.Data)*4) {
			return Encoded{Hash: h, Kind: KindVDelta, Payload: delta, Depth: baseDepth + 1}
		}
	}
	full := encodeVecFull(s)
	if delta != nil && len(delta) < len(full) {
		return Encoded{Hash: h, Kind: KindVDelta, Payload: delta, Depth: baseDepth + 1}
	}
	return Encoded{Hash: h, Kind: KindVFull, Payload: full}
}

func encodeVecFull(s *VecSeg) []byte {
	w := header(cas.Zero, 0)
	w.Uvarint(uint64(s.Dim))
	w.SortedU64s(s.IDs)
	w.Bytes(codec.Compress(codec.ShuffleFloats(s.Data)))
	return w.B
}

func encodeVecDelta(s, base *VecSeg, baseHash cas.Hash, depth int) []byte {
	var removed, changed, added []uint64
	var xorData, addData []float32
	tmp := make([]float32, s.Dim)
	i, j := 0, 0
	for i < len(s.IDs) || j < len(base.IDs) {
		switch {
		case j >= len(base.IDs) || (i < len(s.IDs) && s.IDs[i] < base.IDs[j]):
			added = append(added, s.IDs[i])
			addData = append(addData, s.Row(i)...)
			i++
		case i >= len(s.IDs) || base.IDs[j] < s.IDs[i]:
			removed = append(removed, base.IDs[j])
			j++
		default:
			if !rowsEqual(s.Row(i), base.Row(j)) {
				changed = append(changed, s.IDs[i])
				codec.XorFloats(tmp, s.Row(i), base.Row(j))
				xorData = append(xorData, tmp...)
			}
			i++
			j++
		}
	}
	w := header(baseHash, depth)
	w.Uvarint(uint64(s.Dim))
	w.SortedU64s(removed)
	w.SortedU64s(changed)
	w.SortedU64s(added)
	w.Bytes(codec.Compress(codec.ShuffleFloats(xorData)))
	w.Bytes(codec.Compress(codec.ShuffleFloats(addData)))
	return w.B
}

func decodeFloats(r *codec.Reader, n int) ([]float32, error) {
	raw, err := codec.Decompress(r.Bytes())
	if err != nil {
		return nil, err
	}
	out := make([]float32, n)
	if err := codec.UnshuffleFloats(raw, out); err != nil {
		return nil, err
	}
	return out, nil
}

// DecodeVec decodes a vector segment payload.
func DecodeVec(kind byte, payload []byte, resolve Resolver) (*VecSeg, error) {
	r := &codec.Reader{B: payload}
	baseHash, _ := readHeader(r)
	dim := int(r.Uvarint())
	if r.Err() != nil || dim <= 0 {
		return nil, codec.ErrCorrupt
	}
	switch kind {
	case KindVFull:
		ids := r.SortedU64s()
		data, err := decodeFloats(r, len(ids)*dim)
		if err != nil {
			return nil, err
		}
		return &VecSeg{Dim: dim, IDs: ids, Data: data}, r.Err()
	case KindVDelta:
		removed, changed, added := r.SortedU64s(), r.SortedU64s(), r.SortedU64s()
		xorData, err := decodeFloats(r, len(changed)*dim)
		if err != nil {
			return nil, err
		}
		addData, err := decodeFloats(r, len(added)*dim)
		if err != nil {
			return nil, err
		}
		if r.Err() != nil {
			return nil, r.Err()
		}
		bi, err := resolve(baseHash)
		if err != nil {
			return nil, err
		}
		base, ok := bi.(*VecSeg)
		if !ok {
			return nil, errors.New("segment: vec delta base has wrong type")
		}
		out := &VecSeg{Dim: dim}
		ri, ci, ai := 0, 0, 0
		tmp := make([]float32, dim)
		emitAdded := func(upto uint64, inclusive bool) {
			for ai < len(added) && (added[ai] < upto || (inclusive && added[ai] <= upto)) {
				out.IDs = append(out.IDs, added[ai])
				out.Data = append(out.Data, addData[ai*dim:(ai+1)*dim]...)
				ai++
			}
		}
		for j, id := range base.IDs {
			emitAdded(id, false)
			if ri < len(removed) && removed[ri] == id {
				ri++
				continue
			}
			row := base.Row(j)
			if ci < len(changed) && changed[ci] == id {
				codec.XorFloats(tmp, xorData[ci*dim:(ci+1)*dim], row)
				row = tmp
				ci++
			}
			out.IDs = append(out.IDs, id)
			out.Data = append(out.Data, row...)
		}
		emitAdded(math.MaxUint64, true)
		if ri != len(removed) || ci != len(changed) || ai != len(added) {
			return nil, errors.New("segment: vec delta does not match its base")
		}
		return out, nil
	}
	return nil, fmt.Errorf("segment: not a vec kind %d", kind)
}

// ===========================================================================
// CodeSeg — lossy residual-PQ codes
// ===========================================================================

// CodeSeg holds, for each id, a fingerprint of the ORIGINAL float vector (so the next
// commit can tell "unchanged" without keeping the original) and its PQ code.
type CodeSeg struct {
	Codebook cas.Hash
	CodeSize int
	IDs      []uint64
	FPs      []uint64
	Codes    []byte // len(IDs)*CodeSize
}

func (s *CodeSeg) Code(i int) []byte { return s.Codes[i*s.CodeSize : (i+1)*s.CodeSize] }

// Fingerprint hashes a vector's exact bits.
func Fingerprint(v []float32) uint64 {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(f))
	}
	return xxhash.Sum64(b)
}

func (s *CodeSeg) Hash() cas.Hash {
	var w codec.Writer
	w.U8('Q')
	w.Raw(s.Codebook[:])
	w.U32(uint32(s.CodeSize))
	w.U32(uint32(len(s.IDs)))
	for i, id := range s.IDs {
		w.U64(id)
		w.U64(s.FPs[i])
	}
	w.Raw(s.Codes)
	return cas.Sum(w.B)
}

func writeCodes(w *codec.Writer, ids, fps []uint64, codes []byte) {
	var inner codec.Writer
	inner.SortedU64s(ids)
	for _, f := range fps {
		inner.U64(f)
	}
	inner.Raw(codes)
	w.Bytes(codec.Compress(inner.B))
}

// EncodeCode encodes a code segment, as a delta when smaller.
func EncodeCode(s *CodeSeg, base *CodeSeg, baseHash cas.Hash, baseDepth int) Encoded {
	h := s.Hash()
	w := header(cas.Zero, 0)
	w.Raw(s.Codebook[:])
	w.Uvarint(uint64(s.CodeSize))
	writeCodes(w, s.IDs, s.FPs, s.Codes)
	best := Encoded{Hash: h, Kind: KindQFull, Payload: w.B}
	if base == nil || base.Codebook != s.Codebook || baseDepth+1 > MaxChain {
		return best
	}
	var removed, upIDs, upFPs []uint64
	var upCodes []byte
	i, j := 0, 0
	for i < len(s.IDs) || j < len(base.IDs) {
		switch {
		case j >= len(base.IDs) || (i < len(s.IDs) && s.IDs[i] < base.IDs[j]):
			upIDs, upFPs, upCodes = append(upIDs, s.IDs[i]), append(upFPs, s.FPs[i]), append(upCodes, s.Code(i)...)
			i++
		case i >= len(s.IDs) || base.IDs[j] < s.IDs[i]:
			removed = append(removed, base.IDs[j])
			j++
		default:
			if s.FPs[i] != base.FPs[j] || !bytes.Equal(s.Code(i), base.Code(j)) {
				upIDs, upFPs, upCodes = append(upIDs, s.IDs[i]), append(upFPs, s.FPs[i]), append(upCodes, s.Code(i)...)
			}
			i++
			j++
		}
	}
	d := header(baseHash, baseDepth+1)
	d.SortedU64s(removed)
	writeCodes(d, upIDs, upFPs, upCodes)
	if len(d.B) < len(best.Payload) {
		best = Encoded{Hash: h, Kind: KindQDelta, Payload: d.B, Depth: baseDepth + 1}
	}
	return best
}

func readCodes(r *codec.Reader, codeSize int) ([]uint64, []uint64, []byte, error) {
	raw, err := codec.Decompress(r.Bytes())
	if err != nil {
		return nil, nil, nil, err
	}
	in := &codec.Reader{B: raw}
	ids := in.SortedU64s()
	fps := make([]uint64, len(ids))
	for i := range fps {
		fps[i] = in.U64()
	}
	codes := append([]byte(nil), in.Raw(len(ids)*codeSize)...)
	return ids, fps, codes, in.Err()
}

// DecodeCode decodes a code segment payload.
func DecodeCode(kind byte, payload []byte, resolve Resolver) (*CodeSeg, error) {
	r := &codec.Reader{B: payload}
	baseHash, _ := readHeader(r)
	switch kind {
	case KindQFull:
		s := &CodeSeg{}
		copy(s.Codebook[:], r.Raw(32))
		s.CodeSize = int(r.Uvarint())
		if r.Err() != nil || s.CodeSize <= 0 {
			return nil, codec.ErrCorrupt
		}
		var err error
		s.IDs, s.FPs, s.Codes, err = readCodes(r, s.CodeSize)
		return s, err
	case KindQDelta:
		bi, err := resolve(baseHash)
		if err != nil {
			return nil, err
		}
		base, ok := bi.(*CodeSeg)
		if !ok {
			return nil, errors.New("segment: code delta base has wrong type")
		}
		removed := r.SortedU64s()
		upIDs, upFPs, upCodes, err := readCodes(r, base.CodeSize)
		if err != nil {
			return nil, err
		}
		out := &CodeSeg{Codebook: base.Codebook, CodeSize: base.CodeSize}
		cs := base.CodeSize
		ri, ui := 0, 0
		emit := func(id, fp uint64, code []byte) {
			out.IDs, out.FPs, out.Codes = append(out.IDs, id), append(out.FPs, fp), append(out.Codes, code...)
		}
		for j, id := range base.IDs {
			for ui < len(upIDs) && upIDs[ui] < id {
				emit(upIDs[ui], upFPs[ui], upCodes[ui*cs:(ui+1)*cs])
				ui++
			}
			if ri < len(removed) && removed[ri] == id {
				ri++
				continue
			}
			if ui < len(upIDs) && upIDs[ui] == id {
				emit(id, upFPs[ui], upCodes[ui*cs:(ui+1)*cs])
				ui++
				continue
			}
			emit(id, base.FPs[j], base.Code(j))
		}
		for ; ui < len(upIDs); ui++ {
			emit(upIDs[ui], upFPs[ui], upCodes[ui*cs:(ui+1)*cs])
		}
		if ri != len(removed) {
			return nil, errors.New("segment: code delta does not match its base")
		}
		return out, nil
	}
	return nil, fmt.Errorf("segment: not a code kind %d", kind)
}

// ===========================================================================
// GraphSeg — HNSW adjacency (TGIS)
// ===========================================================================

// GNode is one node's canonical adjacency.
type GNode struct {
	ID     uint64
	Level  int
	Layers [][]uint64 // sorted neighbour ids per layer, len == Level+1
}

// GraphSeg holds the nodes of one id range, sorted by id.
type GraphSeg struct{ Nodes []GNode }

func (s *GraphSeg) Hash() cas.Hash {
	var w codec.Writer
	w.U8('G')
	w.U32(uint32(len(s.Nodes)))
	for _, n := range s.Nodes {
		w.U64(n.ID)
		w.U8(byte(n.Level))
		for _, l := range n.Layers {
			w.U32(uint32(len(l)))
			for _, x := range l {
				w.U64(x)
			}
		}
	}
	return cas.Sum(w.B)
}

// Edges counts directed edges in the segment.
func (s *GraphSeg) Edges() int {
	e := 0
	for _, n := range s.Nodes {
		for _, l := range n.Layers {
			e += len(l)
		}
	}
	return e
}

func writeNode(w *codec.Writer, n GNode) {
	w.Uvarint(uint64(n.Level))
	for _, l := range n.Layers {
		w.SortedU64s(l)
	}
}

func readNode(r *codec.Reader, id uint64) GNode {
	n := GNode{ID: id, Level: int(r.Uvarint())}
	if n.Level > 64 {
		r.Raw(-1) // force error
		return n
	}
	n.Layers = make([][]uint64, n.Level+1)
	for l := range n.Layers {
		n.Layers[l] = r.SortedU64s()
	}
	return n
}

// diffSorted returns (b \ a, a \ b) for sorted lists.
func diffSorted(a, b []uint64) (add, rem []uint64) {
	i, j := 0, 0
	for i < len(a) || j < len(b) {
		switch {
		case i >= len(a):
			add = append(add, b[j])
			j++
		case j >= len(b):
			rem = append(rem, a[i])
			i++
		case a[i] == b[j]:
			i++
			j++
		case a[i] < b[j]:
			rem = append(rem, a[i])
			i++
		default:
			add = append(add, b[j])
			j++
		}
	}
	return
}

func applySorted(a, add, rem []uint64) []uint64 {
	out := make([]uint64, 0, len(a)+len(add))
	ri := 0
	ai := 0
	for _, x := range a {
		for ai < len(add) && add[ai] < x {
			out = append(out, add[ai])
			ai++
		}
		if ri < len(rem) && rem[ri] == x {
			ri++
			continue
		}
		out = append(out, x)
	}
	return append(out, add[ai:]...)
}

func sameNode(a, b GNode) bool {
	if a.Level != b.Level {
		return false
	}
	for l := range a.Layers {
		if len(a.Layers[l]) != len(b.Layers[l]) {
			return false
		}
		for i := range a.Layers[l] {
			if a.Layers[l][i] != b.Layers[l][i] {
				return false
			}
		}
	}
	return true
}

// GraphDelta summarises edge mutations between two segments: (ΔE+, ΔE−).
type GraphDelta struct {
	NodesAdded, NodesRemoved, NodesModified int
	EdgesAdded, EdgesRemoved                int
}

// DiffGraph computes the TGIS delta statistics between base and cur.
func DiffGraph(base, cur *GraphSeg) GraphDelta {
	var d GraphDelta
	if base == nil {
		base = &GraphSeg{}
	}
	if cur == nil {
		cur = &GraphSeg{}
	}
	i, j := 0, 0
	for i < len(cur.Nodes) || j < len(base.Nodes) {
		switch {
		case j >= len(base.Nodes) || (i < len(cur.Nodes) && cur.Nodes[i].ID < base.Nodes[j].ID):
			d.NodesAdded++
			for _, l := range cur.Nodes[i].Layers {
				d.EdgesAdded += len(l)
			}
			i++
		case i >= len(cur.Nodes) || base.Nodes[j].ID < cur.Nodes[i].ID:
			d.NodesRemoved++
			for _, l := range base.Nodes[j].Layers {
				d.EdgesRemoved += len(l)
			}
			j++
		default:
			a, b := base.Nodes[j], cur.Nodes[i]
			if !sameNode(a, b) {
				d.NodesModified++
				for l := 0; l < max(len(a.Layers), len(b.Layers)); l++ {
					var la, lb []uint64
					if l < len(a.Layers) {
						la = a.Layers[l]
					}
					if l < len(b.Layers) {
						lb = b.Layers[l]
					}
					add, rem := diffSorted(la, lb)
					d.EdgesAdded += len(add)
					d.EdgesRemoved += len(rem)
				}
			}
			i++
			j++
		}
	}
	return d
}

// EncodeGraph encodes a graph segment, as an edge delta against base when smaller.
func EncodeGraph(s *GraphSeg, base *GraphSeg, baseHash cas.Hash, baseDepth int) Encoded {
	h := s.Hash()
	var inner codec.Writer
	ids := make([]uint64, len(s.Nodes))
	for i, n := range s.Nodes {
		ids[i] = n.ID
	}
	inner.SortedU64s(ids)
	for _, n := range s.Nodes {
		writeNode(&inner, n)
	}
	w := header(cas.Zero, 0)
	w.Bytes(codec.Compress(inner.B))
	best := Encoded{Hash: h, Kind: KindGFull, Payload: w.B}
	if base == nil || baseDepth+1 > MaxChain {
		return best
	}
	// delta: removed nodes, added/replaced nodes (full), modified nodes (edge deltas)
	var removed, addedIDs, modIDs []uint64
	var added []GNode
	var mod codec.Writer
	i, j := 0, 0
	for i < len(s.Nodes) || j < len(base.Nodes) {
		switch {
		case j >= len(base.Nodes) || (i < len(s.Nodes) && s.Nodes[i].ID < base.Nodes[j].ID):
			addedIDs, added = append(addedIDs, s.Nodes[i].ID), append(added, s.Nodes[i])
			i++
		case i >= len(s.Nodes) || base.Nodes[j].ID < s.Nodes[i].ID:
			removed = append(removed, base.Nodes[j].ID)
			j++
		default:
			a, b := base.Nodes[j], s.Nodes[i]
			if a.Level != b.Level { // level changed: re-inserted node, store fully
				removed = append(removed, a.ID)
				addedIDs, added = append(addedIDs, b.ID), append(added, b)
			} else if !sameNode(a, b) {
				modIDs = append(modIDs, b.ID)
				for l := range b.Layers {
					add, rem := diffSorted(a.Layers[l], b.Layers[l])
					mod.SortedU64s(add)
					mod.SortedU64s(rem)
				}
			}
			i++
			j++
		}
	}
	var dw codec.Writer
	dw.SortedU64s(removed)
	dw.SortedU64s(addedIDs)
	for _, n := range added {
		writeNode(&dw, n)
	}
	dw.SortedU64s(modIDs)
	dw.Raw(mod.B)
	d := header(baseHash, baseDepth+1)
	d.Bytes(codec.Compress(dw.B))
	if len(d.B) < len(best.Payload) {
		best = Encoded{Hash: h, Kind: KindGDelta, Payload: d.B, Depth: baseDepth + 1}
	}
	return best
}

// DecodeGraph decodes a graph segment payload.
func DecodeGraph(kind byte, payload []byte, resolve Resolver) (*GraphSeg, error) {
	r := &codec.Reader{B: payload}
	baseHash, _ := readHeader(r)
	raw, err := codec.Decompress(r.Bytes())
	if err != nil {
		return nil, err
	}
	in := &codec.Reader{B: raw}
	switch kind {
	case KindGFull:
		ids := in.SortedU64s()
		s := &GraphSeg{Nodes: make([]GNode, len(ids))}
		for i, id := range ids {
			s.Nodes[i] = readNode(in, id)
		}
		return s, in.Err()
	case KindGDelta:
		bi, err := resolve(baseHash)
		if err != nil {
			return nil, err
		}
		base, ok := bi.(*GraphSeg)
		if !ok {
			return nil, errors.New("segment: graph delta base has wrong type")
		}
		removed := in.SortedU64s()
		addedIDs := in.SortedU64s()
		added := make([]GNode, len(addedIDs))
		for i, id := range addedIDs {
			added[i] = readNode(in, id)
		}
		modIDs := in.SortedU64s()
		if in.Err() != nil {
			return nil, in.Err()
		}
		rm := map[uint64]bool{}
		for _, id := range removed {
			rm[id] = true
		}
		out := &GraphSeg{}
		ai, mi := 0, 0
		for _, n := range base.Nodes {
			for ai < len(added) && added[ai].ID < n.ID {
				out.Nodes = append(out.Nodes, added[ai])
				ai++
			}
			if rm[n.ID] {
				if ai < len(added) && added[ai].ID == n.ID {
					out.Nodes = append(out.Nodes, added[ai])
					ai++
				}
				continue
			}
			if mi < len(modIDs) && modIDs[mi] == n.ID {
				nn := GNode{ID: n.ID, Level: n.Level, Layers: make([][]uint64, len(n.Layers))}
				for l := range n.Layers {
					add, rem := in.SortedU64s(), in.SortedU64s()
					nn.Layers[l] = applySorted(n.Layers[l], add, rem)
				}
				n = nn
				mi++
			}
			out.Nodes = append(out.Nodes, n)
		}
		out.Nodes = append(out.Nodes, added[ai:]...)
		if mi != len(modIDs) || in.Err() != nil {
			return nil, errors.New("segment: graph delta does not match its base")
		}
		return out, nil
	}
	return nil, fmt.Errorf("segment: not a graph kind %d", kind)
}
