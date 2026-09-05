package scheduler

import (
	"strconv"
	"strings"
	"time"
)

// cronMatches reports whether a five-field cron expression (minute hour
// day-of-month month day-of-week, Sunday = 0) matches t.
func cronMatches(expr string, t time.Time) bool {
	fields := strings.Fields(expr)
	if len(fields) != 5 {
		return false
	}
	values := [5]int{t.Minute(), t.Hour(), t.Day(), int(t.Month()), int(t.Weekday())}
	for i, v := range values {
		if !matchesField(fields[i], v) {
			return false
		}
	}
	return true
}

// matchesField handles *, n, a-b, */s, a/s and comma-separated lists of those.
func matchesField(expr string, v int) bool {
	for part := range strings.SplitSeq(expr, ",") {
		if part == "*" {
			return true
		}
		if base, step, ok := strings.Cut(part, "/"); ok {
			start, s := 0, 0
			var err error
			if s, err = strconv.Atoi(step); err != nil || s <= 0 {
				continue
			}
			if base != "*" {
				if start, err = strconv.Atoi(base); err != nil {
					continue
				}
			}
			if v >= start && (v-start)%s == 0 {
				return true
			}
			continue
		}
		lo, hi, ok := strings.Cut(part, "-")
		if !ok {
			hi = lo
		}
		a, err1 := strconv.Atoi(lo)
		b, err2 := strconv.Atoi(hi)
		if err1 == nil && err2 == nil && a <= v && v <= b {
			return true
		}
	}
	return false
}
