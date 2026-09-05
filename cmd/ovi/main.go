// Command ovi is the Ovi — Open Voice Assistant server.
package main

import (
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"github.com/bryfur/ovi-voice-assistant/internal/codec"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/console"
	"github.com/bryfur/ovi-voice-assistant/internal/discovery"
	"github.com/bryfur/ovi-voice-assistant/internal/flash"
	"github.com/bryfur/ovi-voice-assistant/internal/setup"
)

type cliFlags struct {
	scan, genKey, setup, flash, debug bool
	sttProvider, ttsProvider          string
	sttModel, ttsModel, agentModel    string
	mcpServers, agents                string
	transport, codec                  string
}

func main() {
	var f cliFlags
	root := &cobra.Command{
		Use:   "ovi [DEVICES...]",
		Short: "Ovi — Open Voice Assistant.",
		Long: "Ovi — Open Voice Assistant.\n\n" +
			"Connect to ESPHome devices by passing DEVICES as IP addresses,\n" +
			"hostnames, or host:port:key triples.",
		Args:          cobra.ArbitraryArgs,
		SilenceUsage:  true,
		SilenceErrors: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			return run(f, args)
		},
	}
	fl := root.Flags()
	fl.BoolVar(&f.scan, "scan", false, "Scan for ESPHome devices on the network.")
	fl.BoolVar(&f.genKey, "gen-key", false, "Generate an encryption key and exit.")
	fl.BoolVar(&f.setup, "setup", false, "Run the interactive setup wizard.")
	fl.BoolVar(&f.flash, "flash", false, "Flash ESPHome firmware to a device.")
	fl.StringVar(&f.sttProvider, "stt-provider", "", "STT provider (nemotron, whisper).")
	fl.StringVar(&f.ttsProvider, "tts-provider", "", "TTS provider (kokoro, piper).")
	fl.StringVar(&f.sttModel, "stt-model", "", "STT model name.")
	fl.StringVar(&f.ttsModel, "tts-model", "", "TTS model name.")
	fl.StringVar(&f.agentModel, "agent-model", "", "LLM model name.")
	fl.StringVar(&f.mcpServers, "mcp-servers", "", `MCP servers JSON: '[{"command": "npx", "args": [...]}]'.`)
	fl.StringVar(&f.agents, "agents", "", "Sub-agents JSON or @path/to/agents.json.")
	fl.StringVar(&f.transport, "transport", "", "Transport: wifi or ble.")
	fl.StringVar(&f.codec, "codec", "", "Audio codec: pcm, lc3, opus.")
	fl.BoolVar(&f.debug, "debug", false, "Enable debug logging.")

	if err := root.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func setupLogging(debug bool) {
	level := slog.LevelInfo
	if debug {
		level = slog.LevelDebug
	}
	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if a.Key == slog.TimeKey {
				return slog.String(slog.TimeKey, a.Value.Time().Format("15:04:05"))
			}
			return a
		},
	})
	slog.SetDefault(slog.New(handler))
}

func run(f cliFlags, deviceArgs []string) error {
	setupLogging(f.debug)

	if f.genKey {
		genKey()
		return nil
	}
	if f.scan {
		return scan()
	}
	if f.flash {
		flash.Run(console.Default())
		return nil
	}
	if f.setup {
		_, err := setup.Run(console.Default(), "")
		return err
	}
	if config.NeedsSetup() && console.IsTerminal() {
		fmt.Print("No configuration found. Running first-time setup...\n\n")
		if _, err := setup.Run(console.Default(), ""); err != nil {
			return err
		}
		fmt.Println()
	}

	// Build overrides — CLI args override nested config.
	overrides := map[string]string{}
	set := func(key, value string) {
		if value != "" {
			overrides[key] = value
		}
	}
	set("stt.provider", f.sttProvider)
	set("stt.model", f.sttModel)
	set("tts.provider", f.ttsProvider)
	set("tts.model", f.ttsModel)
	set("llm.model", f.agentModel)
	set("llm.mcp_servers", f.mcpServers)
	set("llm.agents", f.agents)
	set("transport.type", strings.ToLower(f.transport))
	set("transport.codec", strings.ToLower(f.codec))
	if f.transport != "" && f.transport != "wifi" && f.transport != "ble" {
		return fmt.Errorf("invalid --transport %q (wifi or ble)", f.transport)
	}
	if f.codec != "" {
		if _, err := codec.ParseCodecType(strings.ToLower(f.codec)); err != nil {
			return fmt.Errorf("invalid --codec %q (pcm, lc3, opus)", f.codec)
		}
	}
	settings, err := config.Load(config.LoadOptions{Overrides: overrides})
	if err != nil {
		return err
	}
	return serve(settings, deviceArgs)
}

func genKey() {
	key := flash.GenerateKey()
	path := config.SecretsPath
	if data, err := os.ReadFile(path); err == nil {
		if !strings.Contains(string(data), "api_encryption_key") {
			content := string(data) + fmt.Sprintf("api_encryption_key: %q\n", key)
			if err := os.WriteFile(path, []byte(content), 0o600); err == nil {
				fmt.Printf("Added api_encryption_key to %s\n", path)
			} else {
				fmt.Printf("Failed to write %s: %v\nKey: %s\n", path, err, key)
			}
		} else {
			fmt.Printf("api_encryption_key already exists in %s\n", path)
			fmt.Printf("New key (not saved): %s\n", key)
		}
	} else {
		fmt.Printf("%s not found — key not saved\nKey: %s\n", path, key)
	}
	fmt.Printf("\nConnect with:\n  ovi DEVICE_HOST:6055::%s\n", key)
}

func scan() error {
	fmt.Println("Scanning for ESPHome devices (5s)...")
	devices, err := discovery.DiscoverDevices(5 * time.Second)
	if err != nil {
		return err
	}
	if len(devices) == 0 {
		fmt.Println("No devices found. Make sure your Voice PE is powered on and connected to WiFi.")
		return nil
	}
	raw, _ := config.LoadRaw("")
	existing := map[string]bool{}
	for _, d := range config.RawDevices(raw) {
		existing[strings.SplitN(d, ":", 2)[0]] = true
	}
	fmt.Printf("\nFound %d device(s):\n\n", len(devices))
	var fresh []discovery.Device
	for i, d := range devices {
		tag := ""
		if existing[d.Host] {
			tag = " (configured)"
		} else {
			fresh = append(fresh, d)
		}
		fmt.Printf("  %d. %-28s %-18s %d%s\n", i+1, d.Name, d.IP, d.Port, tag)
	}
	if len(fresh) > 0 && console.IsTerminal() {
		fmt.Println()
		c := console.Default()
		sel := strings.TrimSpace(c.Prompt("Add devices to config (comma-separated numbers, 'all', or Enter to skip)", "", false))
		if sel != "" {
			var hosts []string
			if strings.EqualFold(sel, "all") {
				for _, d := range fresh {
					hosts = append(hosts, d.Host)
				}
			} else {
				for _, s := range strings.Split(sel, ",") {
					var idx int
					if _, err := fmt.Sscanf(strings.TrimSpace(s), "%d", &idx); err == nil &&
						idx >= 1 && idx <= len(devices) && !existing[devices[idx-1].Host] {
						hosts = append(hosts, devices[idx-1].Host)
					}
				}
			}
			if len(hosts) > 0 {
				if err := config.AddDevicesToConfig(hosts, ""); err != nil {
					return err
				}
				for _, h := range hosts {
					fmt.Printf("  Added %s\n", h)
				}
				fmt.Printf("\nConfiguration saved to %s\n", config.ConfigPath())
				return nil
			}
		}
	}
	fmt.Println("\nConnect with:")
	for _, d := range devices {
		fmt.Printf("  ovi %s\n", d.Host)
	}
	return nil
}
