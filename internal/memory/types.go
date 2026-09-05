// Package memory is a lightweight memory system — extract, store, and
// recall facts with SQLite and local embeddings.
package memory

// FactType describes what kind of fact this is.
type FactType string

const (
	FactWorld      FactType = "world"      // Objective info about the world
	FactExperience FactType = "experience" // User's experiences/preferences
	FactAssistant  FactType = "assistant"  // Things the assistant did/said
)

// ParseFactType returns a valid FactType, defaulting to world.
func ParseFactType(s string) FactType {
	switch FactType(s) {
	case FactWorld, FactExperience, FactAssistant:
		return FactType(s)
	}
	return FactWorld
}

// EntityType classifies extracted entities.
type EntityType string

const (
	EntityPerson       EntityType = "person"
	EntityOrganization EntityType = "organization"
	EntityLocation     EntityType = "location"
	EntityConcept      EntityType = "concept"
	EntityOther        EntityType = "other"
)

// ParseEntityType returns a valid EntityType, defaulting to other.
func ParseEntityType(s string) EntityType {
	switch EntityType(s) {
	case EntityPerson, EntityOrganization, EntityLocation, EntityConcept, EntityOther:
		return EntityType(s)
	}
	return EntityOther
}

// Fact is a single extracted fact stored in memory.
type Fact struct {
	ID         string
	BankID     string
	Text       string // Combined fact text for embedding
	What       string // Core fact (1-2 sentences)
	Who        string
	Where      string
	When       string
	Why        string
	FactType   FactType
	Confidence float64
	Embedding  []float32
	CreatedAt  string  // RFC3339 timestamp
	OccurredAt string  // When the fact's event occurred
	Relevance  float64 // Set during recall, not stored
}

// Entity is a named entity linked to facts.
type Entity struct {
	ID         string
	BankID     string
	Text       string // Canonical name
	EntityType EntityType
	Embedding  []float32
	FactIDs    []string
	CreatedAt  string
}

// RetainResult is the outcome of a retain operation.
type RetainResult struct {
	Success    bool
	FactsCount int
	FactIDs    []string
	EntityIDs  []string
}

// RecallResult is the outcome of a recall operation.
type RecallResult struct {
	Results         []Fact
	Entities        []Entity
	TotalCandidates int
}
