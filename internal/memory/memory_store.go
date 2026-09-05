package memory

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	_ "modernc.org/sqlite"
)

const schema = `
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL,
    what TEXT NOT NULL,
    who TEXT DEFAULT '',
    "where" TEXT DEFAULT '',
    "when" TEXT DEFAULT '',
    why TEXT DEFAULT '',
    fact_type TEXT DEFAULT 'world',
    confidence REAL DEFAULT 1.0,
    embedding TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    occurred_at TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_facts_bank ON facts(bank_id);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    bank_id TEXT NOT NULL,
    text TEXT NOT NULL,
    entity_type TEXT DEFAULT 'other',
    embedding TEXT DEFAULT '[]',
    fact_ids TEXT DEFAULT '[]',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_bank ON entities(bank_id);
CREATE INDEX IF NOT EXISTS idx_entities_text ON entities(bank_id, text COLLATE NOCASE);
`

// MemoryStore is a SQLite-backed memory store with in-process vector search.
type MemoryStore struct {
	dbPath string
	db     *sql.DB
}

// NewMemoryStore creates a store for the given path (":memory:" for RAM).
func NewMemoryStore(dbPath string) *MemoryStore {
	return &MemoryStore{dbPath: dbPath}
}

// Open connects and applies the schema.
func (s *MemoryStore) Open() error {
	dsn := s.dbPath
	if dsn != ":memory:" {
		if err := os.MkdirAll(filepath.Dir(dsn), 0o755); err != nil {
			return err
		}
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return err
	}
	// A single connection keeps :memory: databases coherent and avoids
	// SQLite writer contention.
	db.SetMaxOpenConns(1)
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil && dsn != ":memory:" {
		db.Close()
		return err
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return err
	}
	s.db = db
	slog.Info("Memory store opened", "path", s.dbPath)
	return nil
}

// Close closes the database.
func (s *MemoryStore) Close() error {
	if s.db != nil {
		err := s.db.Close()
		s.db = nil
		return err
	}
	return nil
}

func (s *MemoryStore) conn() (*sql.DB, error) {
	if s.db == nil {
		return nil, fmt.Errorf("store not opened")
	}
	return s.db, nil
}

func encodeVec(v []float32) string {
	if len(v) == 0 {
		return "[]"
	}
	b, _ := json.Marshal(v)
	return string(b)
}

func decodeVec(s string) []float32 {
	if s == "" || s == "[]" {
		return nil
	}
	var v []float32
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return nil
	}
	return v
}

func encodeIDs(ids []string) string {
	if ids == nil {
		ids = []string{}
	}
	b, _ := json.Marshal(ids)
	return string(b)
}

func decodeIDs(s string) []string {
	var v []string
	_ = json.Unmarshal([]byte(s), &v)
	return v
}

const factColumns = `id, bank_id, text, what, who, "where", "when", why, fact_type, confidence, embedding, created_at, occurred_at`

func scanFact(rows interface{ Scan(...any) error }) (Fact, error) {
	var f Fact
	var ft, emb string
	err := rows.Scan(&f.ID, &f.BankID, &f.Text, &f.What, &f.Who, &f.Where, &f.When, &f.Why,
		&ft, &f.Confidence, &emb, &f.CreatedAt, &f.OccurredAt)
	if err != nil {
		return f, err
	}
	f.FactType = ParseFactType(ft)
	f.Embedding = decodeVec(emb)
	return f, nil
}

func (s *MemoryStore) queryFacts(query string, args ...any) ([]Fact, error) {
	db, err := s.conn()
	if err != nil {
		return nil, err
	}
	rows, err := db.Query(query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var facts []Fact
	for rows.Next() {
		f, err := scanFact(rows)
		if err != nil {
			return nil, err
		}
		facts = append(facts, f)
	}
	return facts, rows.Err()
}

// -- Facts --

// SaveFacts upserts facts.
func (s *MemoryStore) SaveFacts(facts []Fact) error {
	db, err := s.conn()
	if err != nil {
		return err
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO facts (` + factColumns + `)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, f := range facts {
		ft := f.FactType
		if ft == "" {
			ft = FactWorld
		}
		if _, err := stmt.Exec(f.ID, f.BankID, f.Text, f.What, f.Who, f.Where, f.When, f.Why,
			string(ft), f.Confidence, encodeVec(f.Embedding), f.CreatedAt, f.OccurredAt); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// GetFacts returns facts for a bank, optionally restricted to ids.
func (s *MemoryStore) GetFacts(bankID string, factIDs []string) ([]Fact, error) {
	if len(factIDs) > 0 {
		placeholders := strings.Repeat("?,", len(factIDs))
		placeholders = placeholders[:len(placeholders)-1]
		args := make([]any, 0, len(factIDs)+1)
		args = append(args, bankID)
		for _, id := range factIDs {
			args = append(args, id)
		}
		return s.queryFacts(`SELECT `+factColumns+` FROM facts WHERE bank_id=? AND id IN (`+placeholders+`)`, args...)
	}
	return s.queryFacts(`SELECT `+factColumns+` FROM facts WHERE bank_id=?`, bankID)
}

// ScoredFact pairs a fact with a similarity score.
type ScoredFact struct {
	Fact  Fact
	Score float64
}

// CosineSimilarity computes the cosine similarity of two vectors.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// SearchFactsByEmbedding returns facts ranked by cosine similarity.
func (s *MemoryStore) SearchFactsByEmbedding(bankID string, query []float32, limit int) ([]ScoredFact, error) {
	if len(query) == 0 {
		return nil, nil
	}
	facts, err := s.GetFacts(bankID, nil)
	if err != nil {
		return nil, err
	}
	var qn float64
	for _, v := range query {
		qn += float64(v) * float64(v)
	}
	if qn == 0 {
		return nil, nil
	}
	var scored []ScoredFact
	for _, f := range facts {
		if len(f.Embedding) == 0 || len(f.Embedding) != len(query) {
			continue
		}
		sim := CosineSimilarity(query, f.Embedding)
		scored = append(scored, ScoredFact{Fact: f, Score: sim})
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].Score > scored[j].Score })
	if len(scored) > limit {
		scored = scored[:limit]
	}
	return scored, nil
}

// SearchFactsByText performs a keyword search using LIKE on fact text.
func (s *MemoryStore) SearchFactsByText(bankID, query string, limit int) ([]Fact, error) {
	terms := strings.Fields(strings.ToLower(query))
	if len(terms) == 0 {
		return nil, nil
	}
	conds := make([]string, len(terms))
	args := []any{bankID}
	for i, t := range terms {
		conds[i] = "LOWER(text) LIKE ?"
		args = append(args, "%"+t+"%")
	}
	args = append(args, limit)
	return s.queryFacts(`SELECT `+factColumns+` FROM facts WHERE bank_id=? AND (`+
		strings.Join(conds, " OR ")+`) LIMIT ?`, args...)
}

// -- Entities --

const entityColumns = `id, bank_id, text, entity_type, embedding, fact_ids, created_at`

func scanEntity(rows interface{ Scan(...any) error }) (Entity, error) {
	var e Entity
	var et, emb, ids string
	if err := rows.Scan(&e.ID, &e.BankID, &e.Text, &et, &emb, &ids, &e.CreatedAt); err != nil {
		return e, err
	}
	e.EntityType = ParseEntityType(et)
	e.Embedding = decodeVec(emb)
	e.FactIDs = decodeIDs(ids)
	return e, nil
}

// SaveEntities upserts entities.
func (s *MemoryStore) SaveEntities(entities []Entity) error {
	db, err := s.conn()
	if err != nil {
		return err
	}
	tx, err := db.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()
	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO entities (` + entityColumns + `) VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()
	for _, e := range entities {
		et := e.EntityType
		if et == "" {
			et = EntityOther
		}
		if _, err := stmt.Exec(e.ID, e.BankID, e.Text, string(et), encodeVec(e.Embedding),
			encodeIDs(e.FactIDs), e.CreatedAt); err != nil {
			return err
		}
	}
	return tx.Commit()
}

// GetEntities returns all entities in a bank.
func (s *MemoryStore) GetEntities(bankID string) ([]Entity, error) {
	db, err := s.conn()
	if err != nil {
		return nil, err
	}
	rows, err := db.Query(`SELECT `+entityColumns+` FROM entities WHERE bank_id=?`, bankID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Entity
	for rows.Next() {
		e, err := scanEntity(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, e)
	}
	return out, rows.Err()
}

// GetEntityByText finds an entity by (case-insensitive) name and optional type.
func (s *MemoryStore) GetEntityByText(bankID, text string, entityType EntityType) (*Entity, error) {
	db, err := s.conn()
	if err != nil {
		return nil, err
	}
	var row *sql.Row
	if entityType != "" {
		row = db.QueryRow(`SELECT `+entityColumns+` FROM entities WHERE bank_id=? AND text=? COLLATE NOCASE AND entity_type=?`,
			bankID, text, string(entityType))
	} else {
		row = db.QueryRow(`SELECT `+entityColumns+` FROM entities WHERE bank_id=? AND text=? COLLATE NOCASE`, bankID, text)
	}
	e, err := scanEntity(row)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &e, nil
}

// GetFactsForEntity returns all facts linked to an entity by name.
func (s *MemoryStore) GetFactsForEntity(bankID, entityText string) ([]Fact, error) {
	e, err := s.GetEntityByText(bankID, entityText, "")
	if err != nil || e == nil || len(e.FactIDs) == 0 {
		return nil, err
	}
	return s.GetFacts(bankID, e.FactIDs)
}
