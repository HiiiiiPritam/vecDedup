// Package cas is the content-addressable storage layer.
//
// Layout of a repository directory:
//
//	packs/pack-000001.vdp   append-only immutable object data
//	meta/                   Pebble (RocksDB-style LSM) database:
//	    o/<hash32>  -> object location {pack, offset, length, kind}
//	    r/<name>    -> refs (branches), arbitrary small values
//
// Write protocol (crash safe, no partial commits):
//  1. objects are appended to the current pack and fsync'ed;
//  2. ONE atomic Pebble batch (WAL + fsync) publishes all index entries and ref
//     updates together.
//
// If the process dies between 1 and 2 the appended bytes are unreachable garbage
// and the repository is still consistent at the previous commit.
package cas

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"hash/crc32"
	"io/fs"
	"os"
	"path/filepath"
	"sync"

	"github.com/cockroachdb/pebble"
)

// Hash is a SHA-256 content address.
type Hash [32]byte

// Zero is the all-zero hash (used as "no object").
var Zero Hash

func (h Hash) String() string { return hex.EncodeToString(h[:]) }
func (h Hash) Short() string  { return hex.EncodeToString(h[:6]) }
func (h Hash) IsZero() bool   { return h == Zero }

// Sum hashes data.
func Sum(data ...[]byte) Hash {
	s := sha256.New()
	for _, d := range data {
		s.Write(d)
	}
	var h Hash
	s.Sum(h[:0])
	return h
}

// ParseHash parses a hex hash (full length).
func ParseHash(s string) (Hash, error) {
	var h Hash
	b, err := hex.DecodeString(s)
	if err != nil || len(b) != 32 {
		return h, fmt.Errorf("cas: bad hash %q", s)
	}
	copy(h[:], b)
	return h, nil
}

// ErrNotFound is returned for missing objects/keys.
var ErrNotFound = errors.New("cas: not found")

const (
	maxPackSize = 1 << 30
	recHeader   = 32 + 1 + 4 // hash, kind, length
)

type loc struct {
	pack   uint32
	off    int64
	length uint32
	kind   byte
}

func (l loc) encode() []byte {
	b := make([]byte, 17)
	binary.LittleEndian.PutUint32(b, l.pack)
	binary.LittleEndian.PutUint64(b[4:], uint64(l.off))
	binary.LittleEndian.PutUint32(b[12:], l.length)
	b[16] = l.kind
	return b
}

func decodeLoc(b []byte) (loc, error) {
	if len(b) != 17 {
		return loc{}, errors.New("cas: bad index entry")
	}
	return loc{binary.LittleEndian.Uint32(b), int64(binary.LittleEndian.Uint64(b[4:])), binary.LittleEndian.Uint32(b[12:]), b[16]}, nil
}

// Store is a repository's object store + metadata DB.
type Store struct {
	dir string
	db  *pebble.DB

	mu      sync.Mutex // guards writer state and reader cache
	cur     *os.File
	curID   uint32
	curSize int64
	readers map[uint32]*os.File
}

// Open opens (creating if needed) a store in dir.
func Open(dir string) (*Store, error) {
	if err := os.MkdirAll(filepath.Join(dir, "packs"), 0o755); err != nil {
		return nil, err
	}
	db, err := pebble.Open(filepath.Join(dir, "meta"), &pebble.Options{Logger: quietLogger{}})
	if err != nil {
		return nil, err
	}
	s := &Store{dir: dir, db: db, readers: map[uint32]*os.File{}}
	// find the highest pack id
	ents, _ := os.ReadDir(filepath.Join(dir, "packs"))
	for _, e := range ents {
		var id uint32
		if _, err := fmt.Sscanf(e.Name(), "pack-%06d.vdp", &id); err == nil && id > s.curID {
			s.curID = id
		}
	}
	return s, nil
}

type quietLogger struct{}

func (quietLogger) Infof(string, ...interface{})  {}
func (quietLogger) Errorf(f string, a ...interface{}) { fmt.Fprintf(os.Stderr, "pebble: "+f+"\n", a...) }
func (quietLogger) Fatalf(f string, a ...interface{}) { panic(fmt.Sprintf(f, a...)) }

// Close closes all files.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, f := range s.readers {
		f.Close()
	}
	if s.cur != nil {
		s.cur.Close()
	}
	return s.db.Close()
}

func (s *Store) packPath(id uint32) string {
	return filepath.Join(s.dir, "packs", fmt.Sprintf("pack-%06d.vdp", id))
}

func objKey(h Hash) []byte { return append([]byte("o/"), h[:]...) }

func (s *Store) lookup(h Hash) (loc, error) {
	v, closer, err := s.db.Get(objKey(h))
	if err == pebble.ErrNotFound {
		return loc{}, ErrNotFound
	}
	if err != nil {
		return loc{}, err
	}
	defer closer.Close()
	return decodeLoc(v)
}

// Has reports whether an object exists (committed).
func (s *Store) Has(h Hash) bool {
	_, err := s.lookup(h)
	return err == nil
}

// Get returns an object's kind and payload, verifying its CRC.
func (s *Store) Get(h Hash) (byte, []byte, error) {
	l, err := s.lookup(h)
	if err != nil {
		return 0, nil, fmt.Errorf("object %s: %w", h.Short(), err)
	}
	f, err := s.reader(l.pack)
	if err != nil {
		return 0, nil, err
	}
	buf := make([]byte, int(l.length)+4)
	if _, err := f.ReadAt(buf, l.off); err != nil {
		return 0, nil, err
	}
	payload := buf[:l.length]
	if crc32.ChecksumIEEE(payload) != binary.LittleEndian.Uint32(buf[l.length:]) {
		return 0, nil, fmt.Errorf("cas: checksum mismatch for %s", h.Short())
	}
	return l.kind, payload, nil
}

func (s *Store) reader(id uint32) (*os.File, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if f, ok := s.readers[id]; ok {
		return f, nil
	}
	f, err := os.Open(s.packPath(id))
	if err != nil {
		return nil, err
	}
	s.readers[id] = f
	return f, nil
}

// GetMeta reads a metadata key.
func (s *Store) GetMeta(key string) ([]byte, error) {
	v, closer, err := s.db.Get([]byte("r/" + key))
	if err == pebble.ErrNotFound {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, err
	}
	defer closer.Close()
	return append([]byte(nil), v...), nil
}

// ListMeta returns all metadata entries whose key starts with prefix.
func (s *Store) ListMeta(prefix string) (map[string][]byte, error) {
	lo := []byte("r/" + prefix)
	hi := append(append([]byte(nil), lo...), 0xff)
	it, err := s.db.NewIter(&pebble.IterOptions{LowerBound: lo, UpperBound: hi})
	if err != nil {
		return nil, err
	}
	defer it.Close()
	out := map[string][]byte{}
	for it.First(); it.Valid(); it.Next() {
		out[string(it.Key()[2:])] = append([]byte(nil), it.Value()...)
	}
	return out, it.Error()
}

// ---------------------------------------------------------------------------
// Transactions
// ---------------------------------------------------------------------------

type pending struct {
	h       Hash
	kind    byte
	payload []byte
}

// Txn stages object writes and metadata updates, published atomically by Commit.
type Txn struct {
	s       *Store
	objs    []pending
	staged  map[Hash]int
	meta    map[string][]byte
	deletes map[string]bool
	// Stats
	NewObjects   int
	NewBytes     int64
	DedupObjects int
}

// Begin starts a transaction. Only one writer should commit at a time (the repo
// layer holds a lock); readers are never blocked.
func (s *Store) Begin() *Txn {
	return &Txn{s: s, staged: map[Hash]int{}, meta: map[string][]byte{}, deletes: map[string]bool{}}
}

// Put stages an object. If an object with the same hash already exists (committed
// or staged) nothing is written: this is the deduplication point. Returns true if
// the object was new.
func (t *Txn) Put(h Hash, kind byte, payload []byte) bool {
	if _, ok := t.staged[h]; ok {
		t.DedupObjects++
		return false
	}
	if t.s.Has(h) {
		t.DedupObjects++
		return false
	}
	t.staged[h] = len(t.objs)
	t.objs = append(t.objs, pending{h, kind, payload})
	t.NewObjects++
	t.NewBytes += int64(len(payload))
	return true
}

// Has reports whether h is committed or staged in this txn.
func (t *Txn) Has(h Hash) bool {
	if _, ok := t.staged[h]; ok {
		return true
	}
	return t.s.Has(h)
}

// Get reads an object, seeing staged writes.
func (t *Txn) Get(h Hash) (byte, []byte, error) {
	if i, ok := t.staged[h]; ok {
		return t.objs[i].kind, t.objs[i].payload, nil
	}
	return t.s.Get(h)
}

// SetMeta stages a metadata write.
func (t *Txn) SetMeta(key string, val []byte) { t.meta[key] = val; delete(t.deletes, key) }

// DeleteMeta stages a metadata delete.
func (t *Txn) DeleteMeta(key string) { t.deletes[key] = true; delete(t.meta, key) }

// Commit makes all staged objects and metadata durable and visible atomically.
func (t *Txn) Commit() error {
	s := t.s
	s.mu.Lock()
	defer s.mu.Unlock()
	batch := s.db.NewBatch()
	defer batch.Close()

	touched := map[*os.File]bool{}
	for _, o := range t.objs {
		recLen := int64(recHeader + len(o.payload) + 4)
		if s.cur == nil || s.curSize+recLen > maxPackSize {
			if err := s.rotate(); err != nil {
				return err
			}
		}
		rec := make([]byte, 0, recLen)
		rec = append(rec, o.h[:]...)
		rec = append(rec, o.kind)
		rec = binary.LittleEndian.AppendUint32(rec, uint32(len(o.payload)))
		rec = append(rec, o.payload...)
		rec = binary.LittleEndian.AppendUint32(rec, crc32.ChecksumIEEE(o.payload))
		if _, err := s.cur.Write(rec); err != nil {
			return err
		}
		l := loc{pack: s.curID, off: s.curSize + recHeader, length: uint32(len(o.payload)), kind: o.kind}
		s.curSize += recLen
		touched[s.cur] = true
		if err := batch.Set(objKey(o.h), l.encode(), nil); err != nil {
			return err
		}
	}
	// 1. object data durable
	for f := range touched {
		if err := f.Sync(); err != nil {
			return err
		}
	}
	// 2. publish index + refs atomically
	for k, v := range t.meta {
		if err := batch.Set([]byte("r/"+k), v, nil); err != nil {
			return err
		}
	}
	for k := range t.deletes {
		if err := batch.Delete([]byte("r/"+k), nil); err != nil {
			return err
		}
	}
	return batch.Commit(pebble.Sync)
}

func (s *Store) rotate() error {
	if s.cur != nil {
		if err := s.cur.Sync(); err != nil {
			return err
		}
		s.cur.Close()
	}
	// Continue appending to an existing, not-full pack after reopen.
	if s.cur == nil && s.curID > 0 {
		if st, err := os.Stat(s.packPath(s.curID)); err == nil && st.Size() < maxPackSize/2 {
			f, err := os.OpenFile(s.packPath(s.curID), os.O_WRONLY|os.O_APPEND, 0o644)
			if err == nil {
				s.cur, s.curSize = f, st.Size()
				return nil
			}
		}
	}
	s.curID++
	f, err := os.OpenFile(s.packPath(s.curID), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		return err
	}
	s.cur, s.curSize = f, 0
	return nil
}

// Stats reports physical space usage.
type Stats struct {
	PackBytes int64 // object data on disk
	MetaBytes int64 // Pebble index + refs on disk
	Objects   int
}

// Stats walks the repository and returns physical sizes.
func (s *Store) Stats() (Stats, error) {
	var st Stats
	if err := s.db.Flush(); err != nil {
		return st, err
	}
	err := filepath.WalkDir(s.dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return err
		}
		info, err := d.Info()
		if err != nil {
			return err
		}
		if filepath.Ext(p) == ".vdp" {
			st.PackBytes += info.Size()
		} else {
			st.MetaBytes += info.Size()
		}
		return nil
	})
	it, err2 := s.db.NewIter(&pebble.IterOptions{LowerBound: []byte("o/"), UpperBound: []byte("o0")})
	if err2 == nil {
		for it.First(); it.Valid(); it.Next() {
			st.Objects++
		}
		it.Close()
	}
	return st, err
}
