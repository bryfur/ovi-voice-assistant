package cli

import (
	"strings"
	"testing"
)

func TestPromptReturnsDefaultOnEnter(t *testing.T) {
	c, out := scripted("", "value")

	first := c.Prompt("Name", "def", true)
	second := c.Prompt("Name", "def", true)

	if first != "def" || second != "value" {
		t.Fatalf("got %q, %q", first, second)
	}
	if !strings.Contains(out.String(), "Name [def]:") {
		t.Fatalf("prompt not shown: %s", out.String())
	}
}

func TestPromptHiddenFallsBackToInput(t *testing.T) {
	c, _ := scripted("secret")

	got := c.PromptHidden("Key", "")

	if got != "secret" {
		t.Fatalf("got %q", got)
	}
}

func TestConfirm(t *testing.T) {
	c, _ := scripted("", "y", "no")

	if !c.Confirm("Q", true) || !c.Confirm("Q", false) || c.Confirm("Q", true) {
		t.Fatal("confirm semantics wrong")
	}
}

func TestChoiceRetriesUntilValid(t *testing.T) {
	c, out := scripted("bogus", "ble")

	got := c.Choice("Transport", []string{"wifi", "ble"}, "wifi")

	if got != "ble" || !strings.Contains(out.String(), "Enter one of") {
		t.Fatalf("got %q; out=%s", got, out.String())
	}
}

func TestPickByNumberKeyAndDefault(t *testing.T) {
	opts := []Option{{"lc3", "low latency"}, {"opus", "high quality"}, {"pcm", "raw"}}
	c, _ := scripted("2", "pcm", "", "9", "1")

	byNum := c.Pick("Codec:", opts, "lc3")
	byKey := c.Pick("Codec:", opts, "lc3")
	byDefault := c.Pick("Codec:", opts, "lc3")
	afterRetry := c.Pick("Codec:", opts, "lc3")

	if byNum != "opus" || byKey != "pcm" || byDefault != "lc3" || afterRetry != "lc3" {
		t.Fatalf("got %q %q %q %q", byNum, byKey, byDefault, afterRetry)
	}
}

func TestPanelAndRuleRender(t *testing.T) {
	c, out := scripted("")

	c.Panel("Title", "Sub")
	c.Rule("Section")

	if !strings.Contains(out.String(), "│ Title │") || !strings.Contains(out.String(), "── Section") {
		t.Fatalf("output = %s", out.String())
	}
}
