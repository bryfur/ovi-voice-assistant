package agent

import "testing"

func TestParseArgs(t *testing.T) {
	args, err := parseArgs(`{"a": 1, "b": "x", "c": true}`)

	if err != nil || args.Int("a", 0) != 1 || args.String("b", "") != "x" || !args.Bool("c", false) {
		t.Fatalf("got %v, %v", args, err)
	}
	if args.String("a", "") != "1" || args.Float("missing", 2.5) != 2.5 || args.String("missing", "d") != "d" || !args.Bool("missing", true) {
		t.Fatal("defaults and coercions wrong")
	}
	if empty, err := parseArgs(" "); err != nil || len(empty) != 0 {
		t.Fatalf("empty: %v, %v", empty, err)
	}
	if _, err := parseArgs("{bad"); err == nil {
		t.Fatal("expected error")
	}
}

func TestToolDefDefaultsParameters(t *testing.T) {
	def := Tool{Name: "x", Description: "d"}.Def()

	if def.OfFunction == nil || def.OfFunction.Function.Name != "x" || def.OfFunction.Function.Parameters["type"] != "object" {
		t.Fatalf("got %+v", def)
	}
}
