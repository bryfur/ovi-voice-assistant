package memory

import (
	"context"
	"log/slog"
	"math"
	"sort"
	"strings"
	"time"
)

// rrfK is the Reciprocal Rank Fusion constant — standard value from the literature.
const rrfK = 60

// temporalHalfLife: e^(-days/HALF_LIFE). At 30 days, score ~0.37.
const temporalHalfLife = 30.0

// Budget controls how many candidates to retrieve.
type Budget string

const (
	BudgetLow  Budget = "low"  // 50 candidates — fast
	BudgetMid  Budget = "mid"  // 150 candidates — balanced
	BudgetHigh Budget = "high" // 400 candidates — thorough
)

var budgetLimits = map[Budget]int{BudgetLow: 50, BudgetMid: 150, BudgetHigh: 400}

// rankedID is a fact id with a strategy score.
type rankedID struct {
	id    string
	score float64
}

// Recall searches memory using 4 parallel strategies + RRF fusion.
//
// Strategies:
//  1. Semantic — cosine similarity on embeddings
//  2. Keyword (BM25-lite) — LIKE query on fact text
//  3. Entity graph — facts linked to entities mentioned in query
//  4. Temporal — exponential decay favoring recent facts
//
// Results are merged via Reciprocal Rank Fusion and trimmed to token budget.
func Recall(ctx context.Context, bankID, query string, embedder Embedder, store *MemoryStore, budget Budget, maxTokens int) (RecallResult, error) {
	limit, ok := budgetLimits[budget]
	if !ok {
		limit = 150
	}
	if maxTokens <= 0 {
		maxTokens = 2048
	}

	// 1. Semantic search
	var semantic []rankedID
	queryEmb, err := embedder.EmbedOne(ctx, query)
	if err != nil {
		slog.Error("Query embedding failed", "err", err)
		queryEmb = nil
	}
	if len(queryEmb) > 0 {
		scored, err := store.SearchFactsByEmbedding(bankID, queryEmb, limit)
		if err != nil {
			return RecallResult{}, err
		}
		for _, s := range scored {
			semantic = append(semantic, rankedID{s.Fact.ID, s.Score})
		}
	}

	// 2. Keyword search
	keywordFacts, err := store.SearchFactsByText(bankID, query, limit)
	if err != nil {
		return RecallResult{}, err
	}
	keyword := make([]rankedID, 0, len(keywordFacts))
	for _, f := range keywordFacts {
		keyword = append(keyword, rankedID{f.ID, 1.0})
	}

	// 3. Entity graph search
	entities, err := store.GetEntities(bankID)
	if err != nil {
		return RecallResult{}, err
	}
	entityHits := entityGraphSearch(query, entities, limit)

	// 4. Temporal search
	allFacts, err := store.GetFacts(bankID, nil)
	if err != nil {
		return RecallResult{}, err
	}
	temporal := temporalSearch(allFacts, limit, time.Now().UTC())

	rrf := rrfMerge(semantic, keyword, entityHits, temporal)
	if len(rrf) == 0 {
		return RecallResult{Results: []Fact{}, TotalCandidates: 0}, nil
	}

	rankedIDs := make([]string, 0, len(rrf))
	for id := range rrf {
		rankedIDs = append(rankedIDs, id)
	}
	sort.SliceStable(rankedIDs, func(i, j int) bool {
		if rrf[rankedIDs[i]] == rrf[rankedIDs[j]] {
			return rankedIDs[i] < rankedIDs[j]
		}
		return rrf[rankedIDs[i]] > rrf[rankedIDs[j]]
	})

	fetch := rankedIDs
	if len(fetch) > limit {
		fetch = fetch[:limit]
	}
	facts, err := store.GetFacts(bankID, fetch)
	if err != nil {
		return RecallResult{}, err
	}
	factMap := make(map[string]Fact, len(facts))
	for _, f := range facts {
		factMap[f.ID] = f
	}
	ranked := make([]Fact, 0, len(fetch))
	for _, id := range rankedIDs {
		if f, ok := factMap[id]; ok {
			f.Relevance = rrf[id]
			ranked = append(ranked, f)
		}
	}

	return RecallResult{
		Results:         applyTokenBudget(ranked, maxTokens),
		Entities:        gatherEntities(query, entities),
		TotalCandidates: len(rrf),
	}, nil
}

// entityGraphSearch finds facts linked to entities mentioned in the query.
func entityGraphSearch(query string, entities []Entity, limit int) []rankedID {
	if len(entities) == 0 {
		return nil
	}
	q := strings.ToLower(query)
	var results []rankedID
	seen := map[string]bool{}
	for _, e := range entities {
		if !strings.Contains(q, strings.ToLower(e.Text)) {
			continue
		}
		for _, fid := range e.FactIDs {
			if seen[fid] {
				continue
			}
			seen[fid] = true
			results = append(results, rankedID{fid, 1.0})
			if len(results) >= limit {
				return results
			}
		}
	}
	return results
}

// temporalSearch scores all facts by recency using exponential decay.
func temporalSearch(facts []Fact, limit int, now time.Time) []rankedID {
	var scored []rankedID
	for _, f := range facts {
		if f.CreatedAt == "" {
			continue
		}
		created, err := parseTime(f.CreatedAt)
		if err != nil {
			continue
		}
		days := now.Sub(created).Hours() / 24.0
		scored = append(scored, rankedID{f.ID, math.Exp(-days / temporalHalfLife)})
	}
	sort.SliceStable(scored, func(i, j int) bool { return scored[i].score > scored[j].score })
	if len(scored) > limit {
		scored = scored[:limit]
	}
	return scored
}

func parseTime(s string) (time.Time, error) {
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02T15:04:05", "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t, nil
		}
	}
	return time.Time{}, &time.ParseError{Value: s}
}

// rrfMerge implements Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank_i(d))).
func rrfMerge(lists ...[]rankedID) map[string]float64 {
	scores := map[string]float64{}
	for _, list := range lists {
		for rank, item := range list {
			scores[item.id] += 1.0 / float64(rrfK+rank+1)
		}
	}
	return scores
}

// applyTokenBudget keeps facts until the token budget is exceeded (~4 chars/token).
func applyTokenBudget(facts []Fact, maxTokens int) []Fact {
	result := []Fact{}
	tokens := 0
	for _, f := range facts {
		ft := len(f.Text) / 4
		if tokens+ft > maxTokens {
			break
		}
		result = append(result, f)
		tokens += ft
	}
	return result
}

// gatherEntities returns entities mentioned in the query.
func gatherEntities(query string, entities []Entity) []Entity {
	q := strings.ToLower(query)
	out := []Entity{}
	for _, e := range entities {
		if strings.Contains(q, strings.ToLower(e.Text)) {
			out = append(out, e)
		}
	}
	return out
}
