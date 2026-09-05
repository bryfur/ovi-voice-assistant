// Package cli holds the interactive parts of the ovi command: terminal
// prompts, mDNS device discovery, config-file editing, the ESPHome flash
// flow and the setup wizard.
package cli

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"golang.org/x/term"
)

// Console asks questions on a terminal; tests substitute the streams.
type Console struct {
	In     io.Reader
	Out    io.Writer
	Hidden func() (string, error) // reads a line without echo; nil falls back to In
	reader *bufio.Reader
}

// Stdio is the real terminal, with hidden input when stdin is a TTY.
func Stdio() *Console {
	c := &Console{In: os.Stdin, Out: os.Stdout}
	if IsTerminal() {
		c.Hidden = func() (string, error) {
			b, err := term.ReadPassword(int(os.Stdin.Fd()))
			fmt.Fprintln(os.Stdout)
			return string(b), err
		}
	}
	return c
}

// IsTerminal reports whether stdin is a TTY.
func IsTerminal() bool { return term.IsTerminal(int(os.Stdin.Fd())) }

func (c *Console) readLine() (string, error) {
	if c.reader == nil {
		c.reader = bufio.NewReader(c.In)
	}
	line, err := c.reader.ReadString('\n')
	if err != nil && line == "" {
		return "", err
	}
	return strings.TrimSpace(line), nil
}

func (c *Console) Print(format string, args ...any) { fmt.Fprintf(c.Out, format, args...) }
func (c *Console) Println(args ...any)              { fmt.Fprintln(c.Out, args...) }

// Rule prints a section heading.
func (c *Console) Rule(title string) {
	fmt.Fprintf(c.Out, "\n── %s %s\n", title, strings.Repeat("─", max(0, 60-len(title))))
}

// Panel prints lines in a box.
func (c *Console) Panel(lines ...string) {
	width := 0
	for _, l := range lines {
		width = max(width, len([]rune(l)))
	}
	fmt.Fprintf(c.Out, "╭%s╮\n", strings.Repeat("─", width+2))
	for _, l := range lines {
		fmt.Fprintf(c.Out, "│ %-*s │\n", width, l)
	}
	fmt.Fprintf(c.Out, "╰%s╯\n", strings.Repeat("─", width+2))
}

// Prompt asks for a line, returning def when the user just presses Enter.
func (c *Console) Prompt(label, def string, showDefault bool) string {
	if showDefault && def != "" {
		fmt.Fprintf(c.Out, "%s [%s]: ", label, def)
	} else {
		fmt.Fprintf(c.Out, "%s: ", label)
	}
	line, err := c.readLine()
	if err != nil || line == "" {
		return def
	}
	return line
}

// PromptHidden asks for a secret without echo.
func (c *Console) PromptHidden(label, def string) string {
	fmt.Fprintf(c.Out, "%s: ", label)
	read := c.readLine
	if c.Hidden != nil {
		read = c.Hidden
	}
	line, err := read()
	if line = strings.TrimSpace(line); err != nil || line == "" {
		return def
	}
	return line
}

// Confirm asks a yes/no question.
func (c *Console) Confirm(label string, def bool) bool {
	hint := "y/N"
	if def {
		hint = "Y/n"
	}
	fmt.Fprintf(c.Out, "%s [%s]: ", label, hint)
	line, err := c.readLine()
	if err != nil || line == "" {
		return def
	}
	line = strings.ToLower(line)
	return line == "y" || line == "yes"
}

// Choice asks for one of the options, by name.
func (c *Console) Choice(label string, options []string, def string) string {
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

// Option is a menu entry.
type Option struct {
	Key  string
	Desc string
}

// Pick shows a numbered menu and returns the chosen key.
func (c *Console) Pick(label string, options []Option, def string) string {
	fmt.Fprintf(c.Out, "\n  %s\n", label)
	defNum := ""
	for i, o := range options {
		marker := ""
		if o.Key == def {
			marker = " (default)"
			defNum = strconv.Itoa(i + 1)
		}
		fmt.Fprintf(c.Out, "    %d. %s — %s%s\n", i+1, o.Key, o.Desc, marker)
	}
	for {
		raw := c.Prompt("  Choice", defNum, defNum != "")
		if raw == "" {
			return def
		}
		if n, err := strconv.Atoi(raw); err == nil && n >= 1 && n <= len(options) {
			return options[n-1].Key
		}
		for _, o := range options {
			if raw == o.Key {
				return o.Key
			}
		}
		fmt.Fprintf(c.Out, "  Enter a number 1-%d\n", len(options))
	}
}
