package scheduler

import (
	"strconv"
	"strings"
	"time"
)

// matchesField checks if a single cron field matches a value.
func matchesField(expr string, value int) bool {
	if expr == "*" {
		return true
	}
	for _, part := range strings.Split(expr, ",") {
		if strings.Contains(part, "/") {
			base, stepS, _ := strings.Cut(part, "/")
			step, err := strconv.Atoi(stepS)
			if err != nil || step <= 0 {
				continue
			}
			start := 0
			if base != "*" {
				start, err = strconv.Atoi(base)
				if err != nil {
					continue
				}
			}
			if value-start >= 0 && (value-start)%step == 0 {
				return true
			}
		} else if strings.Contains(part, "-") {
			loS, hiS, _ := strings.Cut(part, "-")
			lo, err1 := strconv.Atoi(loS)
			hi, err2 := strconv.Atoi(hiS)
			if err1 != nil || err2 != nil {
				continue
			}
			if lo <= value && value <= hi {
				return true
			}
		} else {
			n, err := strconv.Atoi(part)
			if err != nil {
				continue
			}
			if n == value {
				return true
			}
		}
	}
	return false
}

// CronMatches checks if a 5-field cron expression matches a time.
//
// Fields: minute hour day-of-month month day-of-week
// Day-of-week: 0 = Sunday, 6 = Saturday (standard cron convention).
func CronMatches(expr string, t time.Time) bool {
	parts := strings.Fields(expr)
	if len(parts) != 5 {
		return false
	}
	return matchesField(parts[0], t.Minute()) &&
		matchesField(parts[1], t.Hour()) &&
		matchesField(parts[2], t.Day()) &&
		matchesField(parts[3], int(t.Month())) &&
		matchesField(parts[4], int(t.Weekday()))
}

// ValidateCron checks that an expression has 5 fields.
func ValidateCron(expr string) bool {
	return len(strings.Fields(expr)) == 5
}
