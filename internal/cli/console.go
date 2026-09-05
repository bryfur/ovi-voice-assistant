// Package cli holds the interactive parts of the ovi command: terminal
// prompts, mDNS device discovery, config-file editing, the ESPHome flash
// flow and the setup wizard.
package cli

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/term"
)

// IO bundles the streams used for prompts; tests may substitute them.
type IO struct {
	In  io.Reader
	Out io.Writer
	// Hidden reads a line without echo; nil falls back to In.
	Hidden func() (string, error)
	reader *bufio.Reader
}

// Stdio uses stdin/stdout with hidden input when attached to a terminal.
func Stdio() *IO {
	c := &IO{In: os.Stdin, Out: os.Stdout}
	if term.IsTerminal(int(os.Stdin.Fd())) {
		c.Hidden = func() (string, error) {
			b, err := term.ReadPassword(int(os.Stdin.Fd()))
			fmt.Fprintln(os.Stdout)
			return string(b), err
		}
	}
	return c
}

// IsTerminal reports whether stdin is a TTY.
func IsTerminal() bool {
	return term.IsTerminal(int(os.Stdin.Fd()))
}

func (c *IO) readLine() (string, error) {
	if c.reader == nil {
		c.reader = bufio.NewReader(c.In)
	}
	line, err := c.reader.ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	return strings.TrimRight(line, "\r\n"), nil
}

// Print writes to the output.
func (c *IO) Print(format string, args ...any) {
	fmt.Fprintf(c.Out, format, args...)
}

// Println writes a line to the output.
func (c *IO) Println(args ...any) {
	fmt.Fprintln(c.Out, args...)
}

// Rule prints a section heading.
func (c *IO) Rule(title string) {
	fmt.Fprintf(c.Out, "\n── %s %s\n", title, strings.Repeat("─", max(0, 60-len(title))))
}

// Panel prints a boxed title.
func (c *IO) Panel(lines ...string) {
	width := 0
	for _, l := range lines {
		if len([]rune(l)) > width {
			width = len([]rune(l))
		}
	}
	fmt.Fprintf(c.Out, "╭%s╮\n", strings.Repeat("─", width+2))
	for _, l := range lines {
		fmt.Fprintf(c.Out, "│ %-*s │\n", width, l)
	}
	fmt.Fprintf(c.Out, "╰%s╯\n", strings.Repeat("─", width+2))
}

// Prompt asks for a line of input, returning def when the user presses Enter.
func (c *IO) Prompt(label, def string, showDefault bool) string {
	if showDefault && def != "" {
		fmt.Fprintf(c.Out, "%s [%s]: ", label, def)
	} else {
		fmt.Fprintf(c.Out, "%s: ", label)
	}
	line, err := c.readLine()
	if err != nil || strings.TrimSpace(line) == "" {
		return def
	}
	return strings.TrimSpace(line)
}

// PromptHidden asks for a secret without echo.
func (c *IO) PromptHidden(label, def string) string {
	fmt.Fprintf(c.Out, "%s: ", label)
	var line string
	var err error
	if c.Hidden != nil {
		line, err = c.Hidden()
	} else {
		line, err = c.readLine()
	}
	if err != nil || strings.TrimSpace(line) == "" {
		return def
	}
	return strings.TrimSpace(line)
}

// Confirm asks a yes/no question.
func (c *IO) Confirm(label string, def bool) bool {
	hint := "y/N"
	if def {
		hint = "Y/n"
	}
	fmt.Fprintf(c.Out, "%s [%s]: ", label, hint)
	line, err := c.readLine()
	if err != nil {
		return def
	}
	switch strings.ToLower(strings.TrimSpace(line)) {
	case "":
		return def
	case "y", "yes":
		return true
	default:
		return false
	}
}

// Choice asks the user to pick one of the given values.
func (c *IO) Choice(label string, options []string, def string) string {
	for {
		v := strings.ToLower(c.Prompt(fmt.Sprintf("%s (%s)", label, strings.Join(options, "/")), def, true))
		for _, o := range options {
			if v == o {
				return o
			}
		}
		fmt.Fprintf(c.Out, "  Enter one of: %s\n", strings.Join(options, ", "))
	}
}

// Option is a labelled menu entry.
type Option struct {
	Key  string
	Desc string
}

// Pick shows numbered options with descriptions and returns the chosen key.
func (c *IO) Pick(label string, options []Option, def string) string {
	fmt.Fprintf(c.Out, "\n  %s\n", label)
	defNum := ""
	for i, o := range options {
		marker := ""
		if o.Key == def {
			marker = " (default)"
			defNum = fmt.Sprint(i + 1)
		}
		fmt.Fprintf(c.Out, "    %d. %s — %s%s\n", i+1, o.Key, o.Desc, marker)
	}
	for {
		raw := c.Prompt("  Choice", defNum, defNum != "")
		if raw == "" {
			return def
		}
		var idx int
		if _, err := fmt.Sscanf(raw, "%d", &idx); err == nil && idx >= 1 && idx <= len(options) {
			return options[idx-1].Key
		}
		for _, o := range options {
			if raw == o.Key {
				return o.Key
			}
		}
		fmt.Fprintf(c.Out, "  Enter a number 1-%d\n", len(options))
	}
}
