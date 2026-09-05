// Command ovi is the Ovi — Open Voice Assistant server.
package main

import (
	"errors"
	"flag"
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/cli"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

const usage = `Ovi — Open Voice Assistant.

Usage: ovi [flags] [DEVICE ...]

Connect to ESPHome devices by passing DEVICEs as IP addresses, hostnames,
or host:port:key triples. Flags may appear anywhere.

Flags:
`

func main() {
	if err := run(os.Args[1:]); err != nil && !errors.Is(err, flag.ErrHelp) {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("ovi", flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(fs.Output(), usage); fs.PrintDefaults() }
	var (
		scan     = fs.Bool("scan", false, "Scan for ESPHome devices on the network.")
		genKey   = fs.Bool("gen-key", false, "Generate an encryption key and exit.")
		setupCmd = fs.Bool("setup", false, "Run the interactive setup wizard.")
		flashCmd = fs.Bool("flash", false, "Flash ESPHome firmware to a device.")
		debug    = fs.Bool("debug", false, "Enable debug logging.")
		verbose  = fs.Bool("verbose", false, "Log every LLM stream chunk (implies --debug).")
	)
	overrides := map[string]string{}
	for key, help := range map[string]string{
		"stt.provider":    "STT provider (nemotron, whisper).",
		"stt.model":       "STT model (nemotron chunk like 560ms, or whisper model like base.en).",
		"tts.provider":    "TTS provider (kokoro, piper).",
		"tts.model":       "TTS voice.",
		"llm.model":       "LLM model name.",
		"llm.mcp_servers": "MCP servers JSON or @path.",
		"llm.agents":      "Sub-agents JSON or @path.",
		"transport.type":  "Transport: wifi or ble.",
		"transport.codec": "Audio codec: pcm, lc3, opus.",
	} {
		flagName := strings.NewReplacer(".", "-", "_", "-").Replace(key)
		fs.Func(flagName, help, func(v string) error { overrides[key] = v; return nil })
	}

	// Allow flags and positional devices to be interspersed.
	var devices []string
	for {
		if err := fs.Parse(args); err != nil {
			return err
		}
		if fs.NArg() == 0 {
			break
		}
		devices = append(devices, fs.Arg(0))
		args = fs.Args()[1:]
	}
	setupLogging(*debug, *verbose)

	switch {
	case *genKey:
		return genKeyCmd()
	case *scan:
		return scanCmd()
	case *flashCmd:
		cli.Flash(cli.Stdio())
		return nil
	case *setupCmd:
		_, err := cli.Setup(cli.Stdio(), "")
		return err
	}
	if cli.NeedsSetup() && cli.IsTerminal() {
		fmt.Print("No configuration found. Running first-time setup...\n\n")
		if _, err := cli.Setup(cli.Stdio(), ""); err != nil {
			return err
		}
		fmt.Println()
	}
	settings, err := config.Load(config.LoadOptions{Overrides: overrides})
	if err != nil {
		return err
	}
	return serve(settings, devices)
}

func setupLogging(debug, verbose bool) {
	level := slog.LevelInfo
	switch {
	case verbose:
		level = agent.LevelTrace
	case debug:
		level = slog.LevelDebug
	}
	slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
		ReplaceAttr: func(_ []string, a slog.Attr) slog.Attr {
			switch {
			case a.Key == slog.TimeKey:
				a.Value = slog.StringValue(a.Value.Time().Format("15:04:05.000"))
			case a.Key == slog.LevelKey && a.Value.Any() == agent.LevelTrace:
				a.Value = slog.StringValue("TRACE")
			}
			return a
		},
	})))
}

func genKeyCmd() error {
	key := cli.GenerateKey()
	path := cli.SecretsPath
	data, err := os.ReadFile(path)
	switch {
	case err != nil:
		fmt.Printf("%s not found — key not saved\nKey: %s\n", path, key)
	case strings.Contains(string(data), "api_encryption_key"):
		fmt.Printf("api_encryption_key already exists in %s\nNew key (not saved): %s\n", path, key)
	default:
		line := fmt.Sprintf("api_encryption_key: %q\n", key)
		if err := os.WriteFile(path, append(data, line...), 0o600); err != nil {
			return err
		}
		fmt.Printf("Added api_encryption_key to %s\n", path)
	}
	fmt.Printf("\nConnect with:\n  ovi DEVICE_HOST:6055::%s\n", key)
	return nil
}

func scanCmd() error {
	fmt.Println("Scanning for ESPHome devices (5s)...")
	devices, err := cli.DiscoverDevices(5 * time.Second)
	if err != nil {
		return err
	}
	if len(devices) == 0 {
		fmt.Println("No devices found. Make sure your device is powered on and connected to WiFi.")
		return nil
	}
	raw, _ := cli.LoadConfigFile("")
	configured := map[string]bool{}
	for _, d := range cli.RawDevices(raw) {
		configured[strings.SplitN(d, ":", 2)[0]] = true
	}
	fmt.Printf("\nFound %d device(s):\n\n", len(devices))
	var fresh []string
	for i, d := range devices {
		tag := ""
		if configured[d.Host] {
			tag = " (configured)"
		} else {
			fresh = append(fresh, d.Host)
		}
		fmt.Printf("  %d. %-28s %-18s %d%s\n", i+1, d.Name, d.IP, d.Port, tag)
	}
	if len(fresh) > 0 && cli.IsTerminal() {
		c := cli.Stdio()
		if c.Confirm("\nAdd the new devices to config?", true) {
			if err := cli.AddDevicesToConfig(fresh, ""); err != nil {
				return err
			}
			fmt.Printf("Added %s\nConfiguration saved to %s\n", strings.Join(fresh, ", "), config.ConfigPath())
			return nil
		}
	}
	fmt.Println("\nConnect with:")
	for _, d := range devices {
		fmt.Printf("  ovi %s\n", d.Host)
	}
	return nil
}
