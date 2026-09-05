package agent

import "testing"

func TestParseArgs(t *testing.T) {
	args, err := ParseArgs(`{"a": 1, "b": "x", "c": true}`)
	if err != nil {
		t.Fatal(err)
	}
	if args.Int("a", 0) != 1 || args.String("b", "") != "x" || !args.Bool("c", false) {
		t.Fatalf("got %v", args)
	}
	if args.String("a", "") != "1" || args.Float("missing", 2.5) != 2.5 || args.String("missing", "d") != "d" {
		t.Fatal("conversions wrong")
	}
}

func TestParseArgsEmptyAndInvalid(t *testing.T) {
	empty, err := ParseArgs("")
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty: %v, %v", empty, err)
	}

	_, err = ParseArgs("{bad")

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestArgsStringCoercions(t *testing.T) {
	args := Args{"f": 1.5, "b": false, "n": nil}

	if args.String("f", "") != "1.5" || args.String("b", "") != "false" || args.String("n", "d") != "d" {
		t.Fatalf("got %q %q %q", args.String("f", ""), args.String("b", ""), args.String("n", "d"))
	}
	yes := Args{"s": "yes"}
	if !args.Bool("s", true) || !yes.Bool("s", false) {
		t.Fatal("bool coercion wrong")
	}
}

func TestToolDefDefaultsParameters(t *testing.T) {
	def := Tool{Name: "x", Description: "d"}.Def()

	if def.Type != "function" || def.Function.Name != "x" || def.Function.Parameters["type"] != "object" {
		t.Fatalf("got %+v", def)
	}
}
