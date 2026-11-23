package cas

import (
	"os"
	"path/filepath"
	"testing"
)

func TestPutGetDedupAndPersistence(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	a, b := []byte("alpha payload"), []byte("beta payload")
	ha, hb := Sum(a), Sum(b)
	tx := s.Begin()
	if !tx.Put(ha, 1, a) || !tx.Put(hb, 2, b) || tx.Put(ha, 1, a) {
		t.Fatal("unexpected Put dedup result inside txn")
	}
	tx.SetMeta("branch/main", ha[:])
	if s.Has(ha) {
		t.Fatal("object visible before commit")
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	tx2 := s.Begin()
	if tx2.Put(ha, 1, a) {
		t.Fatal("committed object was written again")
	}
	s.Close()

	s, err = Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	k, got, err := s.Get(hb)
	if err != nil || k != 2 || string(got) != string(b) {
		t.Fatalf("reopen get: %v %d %q", err, k, got)
	}
	m, _ := s.GetMeta("branch/main")
	if string(m) != string(ha[:]) {
		t.Fatal("meta lost across reopen")
	}
	// appending after reopen must not clobber existing data
	c := []byte("gamma")
	tx3 := s.Begin()
	tx3.Put(Sum(c), 3, c)
	if err := tx3.Commit(); err != nil {
		t.Fatal(err)
	}
	if _, got, _ := s.Get(ha); string(got) != string(a) {
		t.Fatal("old object corrupted by append after reopen")
	}
}

func TestUncommittedTxnLeavesNoTrace(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	p := []byte("never committed")
	tx := s.Begin()
	tx.Put(Sum(p), 1, p)
	tx.SetMeta("branch/x", []byte("y"))
	// simulate a crash: no Commit
	s.Close()
	s, _ = Open(dir)
	defer s.Close()
	if s.Has(Sum(p)) {
		t.Fatal("uncommitted object visible")
	}
	if _, err := s.GetMeta("branch/x"); err == nil {
		t.Fatal("uncommitted ref visible")
	}
}

func TestChecksumDetectsCorruption(t *testing.T) {
	dir := t.TempDir()
	s, _ := Open(dir)
	p := []byte("some important bytes here")
	h := Sum(p)
	tx := s.Begin()
	tx.Put(h, 1, p)
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}
	s.Close()
	path := filepath.Join(dir, "packs", "pack-000001.vdp")
	raw, _ := os.ReadFile(path)
	raw[recHeader+3] ^= 0xff // flip a payload byte
	os.WriteFile(path, raw, 0o644)
	s, _ = Open(dir)
	defer s.Close()
	if _, _, err := s.Get(h); err == nil {
		t.Fatal("corruption not detected")
	}
}
