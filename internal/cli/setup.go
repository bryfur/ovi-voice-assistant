package cli

import (
	"cmp"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/speech/tts"
)

var (
	sttProviders    = []string{"nemotron", "whisper"}
	ttsProviders    = []string{"kokoro", "piper"}
	nemotronOptions = []Option{
		{"560ms", "balanced latency (recommended)"},
		{"160ms", "lowest latency, slightly less accurate"},
		{"1120ms", "most accurate, slower to finish"},
	}
	whisperOptions = []Option{
		{"tiny.en", "fastest, least accurate (English)"},
		{"base.en", "good balance (English, recommended)"},
		{"small.en", "more accurate, slower (English)"},
		{"medium.en", "high accuracy, slow (English)"},
		{"turbo", "large-v3 speed-optimized (multilingual)"},
		{"distil-large-v3", "distilled, fast + accurate"},
	}
	codecOptions = []Option{
		{"lc3", "low latency, good quality (recommended)"},
		{"opus", "high quality, higher latency"},
		{"pcm", "uncompressed, highest bandwidth"},
	}
)

// kokoroVoiceOptions describes the English Kokoro voices.
func kokoroVoiceOptions() []Option {
	accent := map[byte]string{'a': "American", 'b': "British"}
	var out []Option
	for _, v := range tts.KokoroVoices() {
		gender := "Female"
		if v[1] == 'm' {
			gender = "Male"
		}
		out = append(out, Option{v, gender + " " + accent[v[0]]})
	}
	return out
}

// scanDevices and runFlash are replaced by tests.
var (
	scanDevices = func() []Device {
		devices, _ := DiscoverDevices(5 * time.Second)
		return devices
	}
	runFlash = Flash
)

// get reads a nested value from a raw config, or def.
func get(cfg map[string]any, def string, keys ...string) string {
	var node any = cfg
	for _, k := range keys {
		m, _ := node.(map[string]any)
		var ok bool
		if node, ok = m[k]; !ok || node == nil {
			return def
		}
	}
	return fmt.Sprint(node)
}

// selectScannedDevices scans and lets the user add devices to existing.
func selectScannedDevices(c *Console, existing []string) []string {
	c.Println("  Scanning for ESPHome devices...")
	found := scanDevices()
	if len(found) == 0 {
		c.Println("  No devices found on the network.")
		return existing
	}
	c.Print("\n  Found %d device(s):\n", len(found))
	for i, d := range found {
		note := ""
		if slices.ContainsFunc(existing, func(e string) bool { return hostOf(e) == d.Host }) {
			note = " (already configured)"
		}
		c.Print("    %d. %s (%s:%d)%s\n", i+1, d.Name, d.IP, d.Port, note)
	}
	devices := slices.Clone(existing)
	choice := strings.ToLower(c.Prompt("\n  Add devices (comma-separated numbers, 'all', or 'none')", "all", true))
	for _, part := range splitList(choice) {
		if part == "all" {
			for _, d := range found {
				devices = append(devices, d.Host)
			}
		} else if i, err := strconv.Atoi(part); err == nil && i >= 1 && i <= len(found) {
			devices = append(devices, found[i-1].Host)
		}
	}
	return slices.Compact(devices)
}

// Setup drives the interactive wizard. An existing config file supplies
// the defaults. The resulting raw config is returned.
func Setup(c *Console, configPath string) (map[string]any, error) {
	existing, err := LoadConfigFile(configPath)
	if err != nil {
		return nil, err
	}
	editing := len(existing) > 0
	c.Println()
	if editing {
		c.Panel("Ovi — Open Voice Assistant", "Edit configuration — press Enter to keep current values")
	} else {
		c.Panel("Ovi — Open Voice Assistant", "First-time setup wizard")
		c.Print("Press Enter to accept defaults. Type a number to select.\n\n")
	}
	cfg := map[string]any{}

	c.Rule("LLM")
	c.Print("  Ovi uses an LLM for conversation. You can use OpenAI\n  or any compatible API (Ollama, LM Studio, etc).\n\n")
	llm := map[string]any{}
	if v := c.PromptHidden("  API key", get(existing, "", "llm", "api_key")); v != "" {
		llm["api_key"] = v
	}
	if v := c.Prompt("  Base URL (empty for OpenAI)", get(existing, "", "llm", "base_url"), false); v != "" {
		llm["base_url"] = v
	}
	llm["model"] = c.Prompt("  Model", get(existing, "gpt-4o-mini", "llm", "model"), true)
	if c.Confirm("  Disable model reasoning/thinking (faster replies)?", get(existing, "true", "llm", "reasoning") == "false") {
		llm["reasoning"] = false
	}
	cfg["llm"] = llm

	c.Println()
	c.Rule("Speech-to-Text")
	c.Print("  Nemotron — NVIDIA Nemotron Speech 600M, streaming, local (recommended).\n" +
		"  Whisper — OpenAI Whisper, local, decoded after you stop talking.\n\n")
	stt := map[string]any{"provider": c.Choice("  Provider", sttProviders, get(existing, "nemotron", "stt", "provider"))}
	if stt["provider"] == "nemotron" {
		stt["model"] = c.Pick("Nemotron chunk size:", nemotronOptions, get(existing, "560ms", "stt", "model"))
	} else {
		stt["model"] = c.Pick("Whisper model:", whisperOptions, get(existing, "base.en", "stt", "model"))
	}
	cfg["stt"] = stt

	c.Println()
	c.Rule("Text-to-Speech")
	c.Print("  Kokoro — fast, high-quality local TTS.\n  Piper — lighter-weight, lower quality.\n\n")
	speech := map[string]any{"provider": c.Choice("  Provider", ttsProviders, get(existing, "kokoro", "tts", "provider"))}
	if speech["provider"] == "kokoro" {
		speech["model"] = c.Pick("Kokoro voice:", kokoroVoiceOptions(), get(existing, "af_heart", "tts", "model"))
	} else {
		speech["model"] = c.Prompt("  Piper voice model", get(existing, "en_US-lessac-medium", "tts", "model"), true)
	}
	cfg["tts"] = speech

	c.Println()
	c.Rule("Devices")
	c.Print("  Connect to ESPHome devices running the Ovi component.\n\n")
	if c.Confirm("  Flash firmware to a new device?", !editing) {
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
	switch {
	case c.Confirm("  Scan for devices on the network?", true):
		devices = selectScannedDevices(c, current)
	case len(current) > 0:
		devices = splitList(c.Prompt("  Devices (comma-separated, or Enter to keep current)", strings.Join(current, ", "), true))
	default:
		devices = splitList(c.Prompt("  Device address (IP or hostname.local, comma-separated)", "", false))
	}
	if len(devices) > 0 {
		c.Print("  Devices: %s\n", strings.Join(devices, ", "))
		cfg["devices"] = anyList(devices)
	}

	c.Println()
	c.Rule("Transport")
	cfg["transport"] = map[string]any{"codec": c.Pick("Audio codec:", codecOptions, get(existing, "lc3", "transport", "codec"))}

	c.Println()
	c.Rule("Music")
	c.Print("  YouTube Music always works (needs yt-dlp + ffmpeg). Spotify and Apple Music\n  play through a Chromium window you log in to.\n\n")
	var services []string
	for _, name := range []string{"spotify", "apple"} {
		if c.Confirm("  Enable "+name+"?", false) {
			services = append(services, name)
		}
	}
	if len(services) > 0 {
		cfg["music"] = map[string]any{"services": anyList(services)}
	}

	c.Println()
	c.Println("  Configuration Summary")
	for _, section := range []string{"llm", "stt", "tts", "devices", "transport", "music"} {
		switch v := cfg[section].(type) {
		case map[string]any:
			for _, k := range slices.Sorted(func(yield func(string) bool) {
				for k := range v {
					if !yield(k) {
						return
					}
				}
			}) {
				value := fmt.Sprint(v[k])
				if k == "api_key" {
					value = "****"
				}
				c.Print("    %-22s %s\n", section+"."+k, value)
			}
		case []any:
			c.Print("    %-22s %s\n", section, strings.Trim(fmt.Sprint(v), "[]"))
		}
	}
	c.Println()
	if !c.Confirm("  Save configuration?", true) {
		c.Println("\n  Configuration not saved.")
		return cfg, nil
	}
	if err := saveConfigFile(cfg, configPath); err != nil {
		return nil, err
	}
	c.Print("\n  Configuration saved to %s\n", cmp.Or(configPath, config.ConfigPath()))
	c.Println("  Edit the file directly or re-run: ovi --setup")
	c.Println("  Override any value with env vars: OVI_LLM__MODEL=gpt-4o")
	return cfg, nil
}
