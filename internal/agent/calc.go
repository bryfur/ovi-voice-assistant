package agent

import (
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode"
)

// calculate evaluates a math expression supporting + - * / % **, parentheses,
// and the functions sqrt, abs, round, sin, cos, tan, log, log10, log2, ceil,
// floor plus the constants pi and e.
func calculate(expr string) (string, error) {
	p := &calcParser{src: []rune(expr)}
	p.skipSpace()
	if p.eof() {
		return "", errors.New("empty expression")
	}
	v, err := p.parseExpr()
	if err != nil {
		return "", err
	}
	p.skipSpace()
	if !p.eof() {
		return "", fmt.Errorf("unexpected %q", string(p.src[p.pos]))
	}
	return formatNumber(v), nil
}

func formatNumber(v float64) string {
	if math.IsNaN(v) {
		return "nan"
	}
	if math.IsInf(v, 0) {
		if v > 0 {
			return "inf"
		}
		return "-inf"
	}
	if v == math.Trunc(v) && math.Abs(v) < 1e18 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'g', -1, 64)
}

type calcParser struct {
	src []rune
	pos int
}

func (p *calcParser) eof() bool { return p.pos >= len(p.src) }

func (p *calcParser) skipSpace() {
	for !p.eof() && unicode.IsSpace(p.src[p.pos]) {
		p.pos++
	}
}

func (p *calcParser) peek(s string) bool {
	p.skipSpace()
	return strings.HasPrefix(string(p.src[p.pos:]), s)
}

func (p *calcParser) accept(s string) bool {
	if p.peek(s) {
		p.pos += len([]rune(s))
		return true
	}
	return false
}

// expr := term (('+' | '-') term)*
func (p *calcParser) parseExpr() (float64, error) {
	left, err := p.parseTerm()
	if err != nil {
		return 0, err
	}
	for {
		if p.accept("+") {
			r, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left += r
		} else if p.accept("-") {
			r, err := p.parseTerm()
			if err != nil {
				return 0, err
			}
			left -= r
		} else {
			return left, nil
		}
	}
}

// term := unary (('*' | '/' | '//' | '%') unary)*
func (p *calcParser) parseTerm() (float64, error) {
	left, err := p.parseUnary()
	if err != nil {
		return 0, err
	}
	for {
		switch {
		case p.peek("**"):
			return left, nil
		case p.accept("*"):
			r, err := p.parseUnary()
			if err != nil {
				return 0, err
			}
			left *= r
		case p.accept("//"):
			r, err := p.parseUnary()
			if err != nil {
				return 0, err
			}
			if r == 0 {
				return 0, errors.New("division by zero")
			}
			left = math.Floor(left / r)
		case p.accept("/"):
			r, err := p.parseUnary()
			if err != nil {
				return 0, err
			}
			if r == 0 {
				return 0, errors.New("division by zero")
			}
			left /= r
		case p.accept("%"):
			r, err := p.parseUnary()
			if err != nil {
				return 0, err
			}
			if r == 0 {
				return 0, errors.New("modulo by zero")
			}
			m := math.Mod(left, r)
			if m != 0 && (m < 0) != (r < 0) {
				m += r
			}
			left = m
		default:
			return left, nil
		}
	}
}

// unary := ('-' | '+') unary | power
func (p *calcParser) parseUnary() (float64, error) {
	if p.accept("-") {
		v, err := p.parseUnary()
		return -v, err
	}
	if p.accept("+") {
		return p.parseUnary()
	}
	return p.parsePower()
}

// power := atom ('**' unary)?   (right-associative)
func (p *calcParser) parsePower() (float64, error) {
	base, err := p.parseAtom()
	if err != nil {
		return 0, err
	}
	if p.accept("**") {
		exp, err := p.parseUnary()
		if err != nil {
			return 0, err
		}
		return math.Pow(base, exp), nil
	}
	return base, nil
}

func (p *calcParser) parseAtom() (float64, error) {
	p.skipSpace()
	if p.eof() {
		return 0, errors.New("unexpected end of expression")
	}
	c := p.src[p.pos]
	if c == '(' {
		p.pos++
		v, err := p.parseExpr()
		if err != nil {
			return 0, err
		}
		if !p.accept(")") {
			return 0, errors.New("missing closing parenthesis")
		}
		return v, nil
	}
	if unicode.IsDigit(c) || c == '.' {
		return p.parseNumber()
	}
	if unicode.IsLetter(c) || c == '_' {
		return p.parseIdent()
	}
	return 0, fmt.Errorf("unexpected %q", string(c))
}

func (p *calcParser) parseNumber() (float64, error) {
	start := p.pos
	for !p.eof() && (unicode.IsDigit(p.src[p.pos]) || p.src[p.pos] == '.' || p.src[p.pos] == '_') {
		p.pos++
	}
	if !p.eof() && (p.src[p.pos] == 'e' || p.src[p.pos] == 'E') {
		save := p.pos
		p.pos++
		if !p.eof() && (p.src[p.pos] == '+' || p.src[p.pos] == '-') {
			p.pos++
		}
		if p.eof() || !unicode.IsDigit(p.src[p.pos]) {
			p.pos = save
		} else {
			for !p.eof() && unicode.IsDigit(p.src[p.pos]) {
				p.pos++
			}
		}
	}
	text := strings.ReplaceAll(string(p.src[start:p.pos]), "_", "")
	v, err := strconv.ParseFloat(text, 64)
	if err != nil {
		return 0, fmt.Errorf("invalid number %q", text)
	}
	return v, nil
}

var calcConstants = map[string]float64{"pi": math.Pi, "e": math.E}

var calcFunctions = map[string]func(args []float64) (float64, error){
	"sqrt":  unaryFn(math.Sqrt),
	"abs":   unaryFn(math.Abs),
	"sin":   unaryFn(math.Sin),
	"cos":   unaryFn(math.Cos),
	"tan":   unaryFn(math.Tan),
	"log10": unaryFn(math.Log10),
	"log2":  unaryFn(math.Log2),
	"ceil":  unaryFn(math.Ceil),
	"floor": unaryFn(math.Floor),
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

func unaryFn(f func(float64) float64) func([]float64) (float64, error) {
	return func(args []float64) (float64, error) {
		if len(args) != 1 {
			return 0, errors.New("expected exactly 1 argument")
		}
		return f(args[0]), nil
	}
}

func (p *calcParser) parseIdent() (float64, error) {
	start := p.pos
	for !p.eof() && (unicode.IsLetter(p.src[p.pos]) || unicode.IsDigit(p.src[p.pos]) || p.src[p.pos] == '_') {
		p.pos++
	}
	name := string(p.src[start:p.pos])
	if p.accept("(") {
		fn, ok := calcFunctions[name]
		if !ok {
			return 0, fmt.Errorf("name '%s' is not defined", name)
		}
		var args []float64
		if !p.accept(")") {
			for {
				v, err := p.parseExpr()
				if err != nil {
					return 0, err
				}
				args = append(args, v)
				if p.accept(",") {
					continue
				}
				if p.accept(")") {
					break
				}
				return 0, errors.New("missing closing parenthesis")
			}
		}
		return fn(args)
	}
	if v, ok := calcConstants[name]; ok {
		return v, nil
	}
	if p.peek(".") || p.peek("[") {
		return 0, errors.New("attribute access and indexing are not allowed")
	}
	return 0, fmt.Errorf("name '%s' is not defined", name)
}
