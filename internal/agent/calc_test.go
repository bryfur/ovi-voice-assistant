package agent

import "testing"

func TestCalculate(t *testing.T) {
	cases := map[string]string{
		"2 ** 10":          "1024",
		"sqrt(144)":        "12",
		"7 / 2":            "3.5",
		"7 // 2":           "3",
		"10 % 3":           "1",
		"-3 + 5":           "2",
		"2 * (3 + 4)":      "14",
		"abs(-2.5)":        "2.5",
		"round(2.5)":       "2",
		"round(3.14159,2)": "3.14",
		"log(100, 10)":     "2",
		"log10(1000)":      "3",
		"ceil(1.2)":        "2",
		"floor(1.8)":       "1",
		"1e3":              "1000",
		"2 ** 3 ** 2":      "512",
		"1_000 + 1":        "1001",
	}

	for expr, want := range cases {
		got, err := calculate(expr)

		if err != nil || got != want {
			t.Errorf("calculate(%q) = %q, %v; want %q", expr, got, err, want)
		}
	}
}

func TestCalculateTrig(t *testing.T) {
	got, err := calculate("sin(pi/2) + cos(0)")

	if err != nil || got != "2" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestCalculateErrors(t *testing.T) {
	for _, expr := range []string{"", "1 +", "foo(1)", "x.y", "1 / 0", "(1", "import os", "2 3"} {
		_, err := calculate(expr)

		if err == nil {
			t.Errorf("calculate(%q) should fail", expr)
		}
	}
}

func TestFormatNumber(t *testing.T) {
	if formatNumber(1.0/3) != "0.3333333333333333" || formatNumber(5) != "5" {
		t.Fatal("formatting wrong")
	}
}
