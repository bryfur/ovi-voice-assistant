// Package setup is the interactive first-run configuration wizard.
package setup

import (
	"fmt"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/console"
	"github.com/bryfur/ovi-voice-assistant/internal/discovery"
	"github.com/bryfur/ovi-voice-assistant/internal/flash"
)

// Available options for each provider.
var (
	STTProviders   = []string{"nemotron", "whisper"}
	TTSProviders   = []string{"kokoro", "piper"}
	NemotronModels = []console.Option{
		{Key: "int8-dynamic", Desc: "best for CPU (recommended)"},
		{Key: "fp16", Desc: "best for NVIDIA GPU"},
		{Key: "int8-static", Desc: "GPU, less VRAM than fp16"},
		{Key: "fp32", Desc: "original precision, largest"},
	}
	WhisperModels = []console.Option{
		{Key: "whisper-1", Desc: "OpenAI hosted Whisper (recommended with OpenAI)"},
		{Key: "tiny.en", Desc: "fastest, least accurate (English) — local server"},
		{Key: "base.en", Desc: "good balance (English) — local server"},
		{Key: "small.en", Desc: "more accurate, slower (English) — local server"},
		{Key: "medium.en", Desc: "high accuracy, slow (English) — local server"},
		{Key: "large-v3", Desc: "best accuracy, slowest — local server"},
		{Key: "turbo", Desc: "large-v3 speed-optimized — local server"},
		{Key: "distil-large-v3", Desc: "distilled, fast + accurate (English) — local server"},
	}
	KokoroVoices = []console.Option{
		{Key: "af_heart", Desc: "Female American (default)"},
		{Key: "af_bella", Desc: "Female American"},
		{Key: "af_nicole", Desc: "Female American"},
		{Key: "af_sarah", Desc: "Female American"},
		{Key: "af_sky", Desc: "Female American"},
		{Key: "am_adam", Desc: "Male American"},
		{Key: "am_michael", Desc: "Male American"},
		{Key: "bf_emma", Desc: "Female British"},
		{Key: "bf_isabella", Desc: "Female British"},
		{Key: "bm_george", Desc: "Male British"},
		{Key: "bm_lewis", Desc: "Male British"},
	}
	Codecs = []console.Option{
		{Key: "lc3", Desc: "low latency, good quality (recommended)"},
		{Key: "opus", Desc: "high quality, higher latency"},
		{Key: "pcm", Desc: "uncompressed, highest bandwidth"},
	}
)

// Scan is the discovery function; tests may replace it.
var Scan = func() []discovery.Device {
	devices, err := discovery.DiscoverDevices(5 * time.Second)
	if err != nil {
		return nil
	}
	return devices
}

// Flash is the flashing flow; tests may replace it.
var Flash = flash.Run

func get(existing map[string]any, def string, keys ...string) string {
	var node any = existing
	for _, k := range keys {
		m, ok := node.(map[string]any)
		if !ok {
			return def
		}
		node, ok = m[k]
		if !ok || node == nil {
			return def
		}
	}
	if s, ok := node.(string); ok {
		return s
	}
	return fmt.Sprint(node)
}

func selectScannedDevices(c *console.IO, existing []string) []string {
	c.Println("  Scanning for ESPHome devices...")
	found := Scan()
	if len(found) == 0 {
		c.Println("  No devices found on the network.")
		return existing
	}
	c.Print("\n  Found %d device(s):\n", len(found))
	known := map[string]bool{}
	for _, e := range existing {
		known[strings.SplitN(e, ":", 2)[0]] = true
	}
	for i, d := range found {
		already := ""
		if known[d.Host] {
			already = " (already configured)"
		}
		c.Print("    %d. %s (%s:%d)%s\n", i+1, d.Name, d.IP, d.Port, already)
	}
	sel := strings.ToLower(c.Prompt("\n  Add devices (comma-separated numbers, 'all', or 'none')", "all", true))
	if sel == "none" {
		return existing
	}
	var newHosts []string
	if sel == "all" {
		for _, d := range found {
			newHosts = append(newHosts, d.Host)
		}
	} else {
		for _, part := range strings.Split(sel, ",") {
			var idx int
			if _, err := fmt.Sscanf(strings.TrimSpace(part), "%d", &idx); err == nil && idx >= 1 && idx <= len(found) {
				newHosts = append(newHosts, found[idx-1].Host)
			}
		}
	}
	// Merge with existing, dedup
	seen := map[string]bool{}
	var merged []string
	for _, h := range append(append([]string{}, existing...), newHosts...) {
		if !seen[h] {
			seen[h] = true
			merged = append(merged, h)
		}
	}
	return merged
}

// Run drives the interactive setup wizard. If a config file already exists
// its values become the defaults. The resulting nested config is returned.
func Run(c *console.IO, configPath string) (map[string]any, error) {
	existing, err := config.LoadRaw(configPath)
	if err != nil {
		return nil, err
	}
	isEdit := len(existing) > 0

	c.Println()
	if isEdit {
		c.Panel("Ovi — Open Voice Assistant", "Edit configuration — press Enter to keep current values")
	} else {
		c.Panel("Ovi — Open Voice Assistant", "First-time setup wizard")
		c.Print("Press Enter to accept defaults. Type a number to select.\n\n")
	}
	cfg := map[string]any{}

	// ── LLM ──────────────────────────────────────────────────
	c.Rule("LLM")
	c.Print("  Ovi uses an LLM for conversation. You can use OpenAI\n  or any compatible API (Ollama, LM Studio, etc).\n\n")
	llm := map[string]any{}
	if v := c.PromptHidden("  API key", get(existing, "", "llm", "api_key")); v != "" {
		llm["api_key"] = v
	}
	if v := c.Prompt("  Base URL (empty for OpenAI)", get(existing, "", "llm", "base_url"), false); v != "" {
		llm["base_url"] = v
	}
	if v := c.Prompt("  Model", get(existing, "gpt-4o-mini", "llm", "model"), true); v != "" {
		llm["model"] = v
	}
	cfg["llm"] = llm

	// ── STT ──────────────────────────────────────────────────
	c.Println()
	c.Rule("Speech-to-Text")
	c.Print("  Nemotron — NVIDIA Nemotron Speech 600M, streaming RNNT, local (recommended).\n" +
		"  Whisper — any OpenAI-compatible transcription endpoint (OpenAI, faster-whisper-server, whisper.cpp).\n\n")
	stt := map[string]any{}
	stt["provider"] = c.Choice("  Provider", STTProviders, get(existing, "nemotron", "stt", "provider"))
	if stt["provider"] == "nemotron" {
		stt["model"] = c.Pick("Nemotron variant:", NemotronModels, get(existing, "int8-dynamic", "stt", "model"))
	} else {
		stt["model"] = c.Pick("Whisper model:", WhisperModels, get(existing, "whisper-1", "stt", "model"))
		if v := c.Prompt("  Transcription base URL (empty = same as LLM)", get(existing, "", "stt", "base_url"), false); v != "" {
			stt["base_url"] = v
		}
	}
	stt["device"] = c.Choice("  Compute device", []string{"cpu", "cuda"}, get(existing, "cpu", "stt", "device"))
	cfg["stt"] = stt

	// ── TTS ──────────────────────────────────────────────────
	c.Println()
	c.Rule("Text-to-Speech")
	c.Print("  Kokoro — fast, high-quality local TTS.\n  Piper — lighter-weight, lower quality.\n  Both need espeak-ng installed for phonemization.\n\n")
	tts := map[string]any{}
	tts["provider"] = c.Choice("  Provider", TTSProviders, get(existing, "kokoro", "tts", "provider"))
	if tts["provider"] == "kokoro" {
		tts["model"] = c.Pick("Kokoro voice:", KokoroVoices, get(existing, "af_heart", "tts", "model"))
	} else {
		tts["model"] = c.Prompt("  Piper voice model", get(existing, "en_US-lessac-medium", "tts", "model"), true)
	}
	cfg["tts"] = tts

	// ── Devices ──────────────────────────────────────────────
	c.Println()
	c.Rule("Devices")
	c.Print("  Connect to ESPHome devices running the Ovi component.\n\n")
	if c.Confirm("  Flash firmware to a new device?", !isEdit) {
		Flash(c)
		c.Println()
	}
	current := config.RawDevices(existing)
	if len(current) > 0 {
		c.Println("  Current devices:")
		for _, d := range current {
			c.Print("    • %s\n", d)
		}
		c.Println()
	}
	var devices []string
	if c.Confirm("  Scan for devices on the network?", true) {
		devices = selectScannedDevices(c, current)
		if len(devices) > 0 {
			c.Print("  Devices: %s\n", strings.Join(devices, ", "))
		}
	} else if len(current) > 0 {
		raw := c.Prompt("  Devices (comma-separated, or Enter to keep current)", strings.Join(current, ", "), true)
		devices = splitList(raw)
	} else {
		raw := c.Prompt("  Device address (IP or hostname.local, comma-separated)", "", false)
		devices = splitList(raw)
	}
	if len(devices) > 0 {
		list := make([]any, len(devices))
		for i, d := range devices {
			list[i] = d
		}
		cfg["devices"] = list
	}

	// ── Transport ────────────────────────────────────────────
	c.Println()
	c.Rule("Transport")
	cfg["transport"] = map[string]any{
		"codec": c.Pick("Audio codec:", Codecs, get(existing, "lc3", "transport", "codec")),
	}

	// ── Summary & Save ───────────────────────────────────────
	c.Println()
	c.Println("  Configuration Summary")
	for _, section := range []string{"llm", "stt", "tts", "devices", "transport"} {
		switch v := cfg[section].(type) {
		case map[string]any:
			for k, val := range v {
				display := fmt.Sprint(val)
				if k == "api_key" {
					display = "****"
				}
				c.Print("    %-22s %s\n", section+"."+k, display)
			}
		case []any:
			strs := make([]string, len(v))
			for i, item := range v {
				strs[i] = fmt.Sprint(item)
			}
			c.Print("    %-22s %s\n", section, strings.Join(strs, ", "))
		}
	}
	c.Println()
	if c.Confirm("  Save configuration?", true) {
		if err := config.SaveRaw(cfg, configPath); err != nil {
			return nil, err
		}
		path := configPath
		if path == "" {
			path = config.ConfigPath()
		}
		c.Print("\n  Configuration saved to %s\n", path)
		c.Println("  Edit the file directly or re-run: ovi --setup")
		c.Println("  Override any value with env vars: OVI_LLM__MODEL=gpt-4o")
	} else {
		c.Println("\n  Configuration not saved.")
	}
	return cfg, nil
}

func splitList(raw string) []string {
	var out []string
	for _, d := range strings.Split(raw, ",") {
		if d = strings.TrimSpace(d); d != "" {
			out = append(out, d)
		}
	}
	return out
}
