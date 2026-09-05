package scheduler

import (
	"testing"
	"time"
)

func TestMatchesField(t *testing.T) {
	cases := []struct {
		expr  string
		value int
		want  bool
	}{
		{"*", 5, true},
		{"5", 5, true}, {"5", 6, false},
		{"1-5", 3, true}, {"1-5", 6, false},
		{"*/15", 0, true}, {"*/15", 30, true}, {"*/15", 10, false},
		{"10/5", 10, true}, {"10/5", 25, true}, {"10/5", 5, false},
		{"1,3,5", 3, true}, {"1,3,5", 4, false},
		{"1-3,10,*/20", 40, true}, {"1-3,10,*/20", 11, false},
	}

	for _, c := range cases {
		got := matchesField(c.expr, c.value)

		if got != c.want {
			t.Errorf("matchesField(%q, %d) = %v, want %v", c.expr, c.value, got, c.want)
		}
	}
}

func at(y int, mo time.Month, d, h, m int) time.Time {
	return time.Date(y, mo, d, h, m, 0, 0, time.Local)
}

func TestCronMatches(t *testing.T) {
	mon := at(2025, time.January, 6, 7, 0) // Monday
	sun := at(2025, time.January, 5, 7, 0) // Sunday

	if !CronMatches("* * * * *", mon) {
		t.Error("every minute should match")
	}
	if !CronMatches("0 7 * * *", mon) || CronMatches("0 7 * * *", at(2025, time.January, 6, 7, 1)) {
		t.Error("specific time")
	}
	if !CronMatches("0 7 * * 1-5", mon) || CronMatches("0 7 * * 1-5", sun) {
		t.Error("weekday range")
	}
	if !CronMatches("0 7 * * 0", sun) {
		t.Error("sunday = 0")
	}
	if !CronMatches("*/30 * * * *", at(2025, time.January, 6, 7, 30)) {
		t.Error("every 30 minutes")
	}
	if !CronMatches("0 7 * 1 *", mon) || CronMatches("0 7 * 2 *", mon) {
		t.Error("month filter")
	}
	if CronMatches("0 7 * *", mon) {
		t.Error("invalid expression should not match")
	}
}
