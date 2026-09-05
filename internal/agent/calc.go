package agent

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// calculate evaluates arithmetic with + - * / // % **, parentheses, the
// constants pi and e, and the functions sqrt abs round sin cos tan log
// log10 log2 ceil floor. Precedence and associativity follow Python.
func calculate(expr string) (string, error) {
	p := &parser{src: expr}
	if p.skip(); p.done() {
		return "", errors.New("empty expression")
	}
	v, err := p.expr(1)
	if err != nil {
		return "", err
	}
	if p.skip(); !p.done() {
		return "", fmt.Errorf("unexpected %q", p.src[p.pos:p.pos+1])
	}
	return formatNumber(v), nil
}

func formatNumber(v float64) string {
	if v == math.Trunc(v) && math.Abs(v) < 1e18 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

type parser struct {
	src string
	pos int
}

// binary operators by precedence; ** is handled in unary so that -2**2
// is -4 and 2**3**2 groups to the right.
var binaryOps = map[string]struct {
	prec int
	eval func(a, b float64) (float64, error)
}{
	"+":  {1, func(a, b float64) (float64, error) { return a + b, nil }},
	"-":  {1, func(a, b float64) (float64, error) { return a - b, nil }},
	"*":  {2, func(a, b float64) (float64, error) { return a * b, nil }},
	"/":  {2, divide(func(a, b float64) float64 { return a / b })},
	"//": {2, divide(func(a, b float64) float64 { return math.Floor(a / b) })},
	"%":  {2, divide(func(a, b float64) float64 { return a - b*math.Floor(a/b) })},
}

func divide(f func(a, b float64) float64) func(a, b float64) (float64, error) {
	return func(a, b float64) (float64, error) {
		if b == 0 {
			return 0, errors.New("division by zero")
		}
		return f(a, b), nil
	}
}

var constants = map[string]float64{"pi": math.Pi, "e": math.E}

var functions = map[string]func(args []float64) (float64, error){
	"sqrt": unary(math.Sqrt), "abs": unary(math.Abs), "sin": unary(math.Sin), "cos": unary(math.Cos),
	"tan": unary(math.Tan), "log10": unary(math.Log10), "log2": unary(math.Log2),
	"ceil": unary(math.Ceil), "floor": unary(math.Floor),
	"log": func(args []float64) (float64, error) {
		switch len(args) {
		case 1:
			return math.Log(args[0]), nil
		case 2:
			return math.Log(args[0]) / math.Log(args[1]), nil
		}
		return 0, errors.New("log() takes 1 or 2 arguments")
	},
	"round": func(args []float64) (float64, error) {
		switch len(args) {
		case 1:
			return math.RoundToEven(args[0]), nil
		case 2:
			scale := math.Pow(10, args[1])
			return math.RoundToEven(args[0]*scale) / scale, nil
		}
		return 0, errors.New("round() takes 1 or 2 arguments")
	},
}

func unary(f func(float64) float64) func([]float64) (float64, error) {
	return func(args []float64) (float64, error) {
		if len(args) != 1 {
			return 0, errors.New("expected exactly 1 argument")
		}
		return f(args[0]), nil
	}
}

func (p *parser) done() bool { return p.pos >= len(p.src) }

func (p *parser) skip() {
	for !p.done() && p.src[p.pos] == ' ' {
		p.pos++
	}
}

func (p *parser) accept(s string) bool {
	p.skip()
	if strings.HasPrefix(p.src[p.pos:], s) {
		p.pos += len(s)
		return true
	}
	return false
}

// expr parses binary operators of at least minPrec by precedence climbing.
func (p *parser) expr(minPrec int) (float64, error) {
	left, err := p.unary()
	for err == nil {
		p.skip()
		op := ""
		for candidate := range binaryOps {
			if strings.HasPrefix(p.src[p.pos:], candidate) && len(candidate) > len(op) {
				op = candidate
			}
		}
		if op == "" || binaryOps[op].prec < minPrec || strings.HasPrefix(p.src[p.pos:], "**") {
			return left, nil
		}
		p.pos += len(op)
		var right float64
		if right, err = p.expr(binaryOps[op].prec + 1); err == nil {
			left, err = binaryOps[op].eval(left, right)
		}
	}
	return 0, err
}

// unary parses signs and the right-associative power operator.
func (p *parser) unary() (float64, error) {
	if p.accept("-") {
		v, err := p.unary()
		return -v, err
	}
	if p.accept("+") {
		return p.unary()
	}
	base, err := p.atom()
	if err == nil && p.accept("**") {
		var exp float64
		if exp, err = p.unary(); err == nil {
			return math.Pow(base, exp), nil
		}
	}
	return base, err
}

// atom parses a parenthesised expression, a number, a constant or a call.
func (p *parser) atom() (float64, error) {
	if p.accept("(") {
		v, err := p.expr(1)
		if err == nil && !p.accept(")") {
			err = errors.New("missing closing parenthesis")
		}
		return v, err
	}
	if p.done() {
		return 0, errors.New("unexpected end of expression")
	}
	start := p.pos
	switch c := p.src[p.pos]; {
	case isDigit(c) || c == '.':
		for !p.done() && (isDigit(p.src[p.pos]) || strings.ContainsRune("._", rune(p.src[p.pos]))) {
			p.pos++
		}
		if !p.done() && (p.src[p.pos]|0x20) == 'e' { // exponent
			end := p.pos + 1
			if end < len(p.src) && (p.src[end] == '+' || p.src[end] == '-') {
				end++
			}
			for end < len(p.src) && isDigit(p.src[end]) {
				end++
			}
			if isDigit(p.src[end-1]) {
				p.pos = end
			}
		}
		text := strings.ReplaceAll(p.src[start:p.pos], "_", "")
		v, err := strconv.ParseFloat(text, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid number %q", text)
		}
		return v, nil
	case isLetter(c):
		for !p.done() && (isLetter(p.src[p.pos]) || isDigit(p.src[p.pos])) {
			p.pos++
		}
		name := p.src[start:p.pos]
		if !p.accept("(") {
			if v, ok := constants[name]; ok {
				return v, nil
			}
			return 0, fmt.Errorf("name '%s' is not defined", name)
		}
		fn, ok := functions[name]
		if !ok {
			return 0, fmt.Errorf("name '%s' is not defined", name)
		}
		var args []float64
		for !p.accept(")") {
			if len(args) > 0 && !p.accept(",") {
				return 0, errors.New("missing closing parenthesis")
			}
			v, err := p.expr(1)
			if err != nil {
				return 0, err
			}
			args = append(args, v)
		}
		return fn(args)
	default:
		return 0, fmt.Errorf("unexpected %q", string(c))
	}
}

func isDigit(c byte) bool  { return '0' <= c && c <= '9' }
func isLetter(c byte) bool { return c == '_' || 'a' <= c|0x20 && c|0x20 <= 'z' }
