package cli

import (
	"fmt"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/tts"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// Available options for each provider.
var (
	sttProviders    = []string{"nemotron", "whisper"}
	ttsProviders    = []string{"kokoro", "piper"}
	nemotronOptions = []Option{
		{Key: "560ms", Desc: "balanced latency (recommended)"},
		{Key: "160ms", Desc: "lowest latency, slightly less accurate"},
		{Key: "1120ms", Desc: "most accurate, slower to finish"},
	}
	whisperOptions = []Option{
		{Key: "tiny.en", Desc: "fastest, least accurate (English)"},
		{Key: "base.en", Desc: "good balance (English, recommended)"},
		{Key: "small.en", Desc: "more accurate, slower (English)"},
		{Key: "medium.en", Desc: "high accuracy, slow (English)"},
		{Key: "turbo", Desc: "large-v3 speed-optimized (multilingual)"},
		{Key: "distil-large-v3", Desc: "distilled, fast + accurate"},
	}
	codecOptions = []Option{
		{Key: "lc3", Desc: "low latency, good quality (recommended)"},
		{Key: "opus", Desc: "high quality, higher latency"},
		{Key: "pcm", Desc: "uncompressed, highest bandwidth"},
	}
)

// kokoroVoiceOptions describes the English Kokoro voices.
func kokoroVoiceOptions() []Option {
	desc := map[byte]string{'a': "American", 'b': "British"}
	var out []Option
	for _, v := range tts.KokoroVoices() {
		gender := "Female"
		if v[1] == 'm' {
			gender = "Male"
		}
		out = append(out, Option{Key: v, Desc: gender + " " + desc[v[0]]})
	}
	return out
}

// Scan is the discovery function; tests may replace it.
var scanDevices = func() []Device {
	devices, err := DiscoverDevices(5 * time.Second)
	if err != nil {
		return nil
	}
	return devices
}

// Flash is the flashing flow; tests may replace it.
var runFlash = Flash

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

func selectScannedDevices(c *IO, existing []string) []string {
	c.Println("  Scanning for ESPHome devices...")
	found := scanDevices()
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

// Setup drives the interactive setup wizard. If a config file already exists
// its values become the defaults. The resulting nested config is returned.
func Setup(c *IO, configPath string) (map[string]any, error) {
	existing, err := LoadConfigFile(configPath)
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
	if c.Confirm("  Disable model reasoning/thinking (faster replies)?", get(existing, "true", "llm", "reasoning") == "false") {
		llm["reasoning"] = false
	}
	cfg["llm"] = llm

	// ── STT ──────────────────────────────────────────────────
	c.Println()
	c.Rule("Speech-to-Text")
	c.Print("  Nemotron — NVIDIA Nemotron Speech 600M, streaming, local (recommended).\n" +
		"  Whisper — OpenAI Whisper, local, decoded after you stop talking.\n\n")
	stt := map[string]any{}
	stt["provider"] = c.Choice("  Provider", sttProviders, get(existing, "nemotron", "stt", "provider"))
	if stt["provider"] == "nemotron" {
		stt["model"] = c.Pick("Nemotron chunk size:", nemotronOptions, get(existing, "560ms", "stt", "model"))
	} else {
		stt["model"] = c.Pick("Whisper model:", whisperOptions, get(existing, "base.en", "stt", "model"))
	}
	cfg["stt"] = stt

	// ── TTS ──────────────────────────────────────────────────
	c.Println()
	c.Rule("Text-to-Speech")
	c.Print("  Kokoro — fast, high-quality local TTS.\n  Piper — lighter-weight, lower quality.\n\n")
	ttsCfg := map[string]any{}
	ttsCfg["provider"] = c.Choice("  Provider", ttsProviders, get(existing, "kokoro", "tts", "provider"))
	if ttsCfg["provider"] == "kokoro" {
		ttsCfg["model"] = c.Pick("Kokoro voice:", kokoroVoiceOptions(), get(existing, "af_heart", "tts", "model"))
	} else {
		ttsCfg["model"] = c.Prompt("  Piper voice model", get(existing, "en_US-lessac-medium", "tts", "model"), true)
	}
	cfg["tts"] = ttsCfg

	// ── Devices ──────────────────────────────────────────────
	c.Println()
	c.Rule("Devices")
	c.Print("  Connect to ESPHome devices running the Ovi component.\n\n")
	if c.Confirm("  Flash firmware to a new device?", !isEdit) {
		runFlash(c)
		c.Println()
	}
	current := RawDevices(existing)
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
		"codec": c.Pick("Audio codec:", codecOptions, get(existing, "lc3", "transport", "codec")),
	}

	// ── Music ────────────────────────────────────────────────
	c.Println()
	c.Rule("Music")
	c.Print("  YouTube Music always works (needs yt-dlp + ffmpeg). Spotify and Apple Music\n  play through a Chromium window you log in to.\n\n")
	var services []any
	for _, name := range []string{"spotify", "apple"} {
		if c.Confirm("  Enable "+name+"?", false) {
			services = append(services, name)
		}
	}
	if len(services) > 0 {
		cfg["music"] = map[string]any{"services": services}
	}

	// ── Summary & Save ───────────────────────────────────────
	c.Println()
	c.Println("  Configuration Summary")
	for _, section := range []string{"llm", "stt", "tts", "devices", "transport", "music"} {
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
		if err := saveConfigFile(cfg, configPath); err != nil {
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
