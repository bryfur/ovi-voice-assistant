package cli

import (
	"cmp"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
)

// sectionOrder is how the config file is laid out; other keys follow, sorted.
var sectionOrder = []string{"llm", "stt", "tts", "devices", "transport", "ble", "music", "automations"}

// LoadConfigFile reads the YAML config as a generic map (empty if missing).
func LoadConfigFile(path string) (map[string]any, error) {
	cfg := map[string]any{}
	data, err := os.ReadFile(cmp.Or(path, config.ConfigPath()))
	if os.IsNotExist(err) {
		return cfg, nil
	}
	if err == nil {
		err = yaml.Unmarshal(data, &cfg)
	}
	if cfg == nil {
		cfg = map[string]any{}
	}
	return cfg, err
}

// saveConfigFile writes the config as YAML, one top-level section per block.
func saveConfigFile(cfg map[string]any, path string) error {
	path = cmp.Or(path, config.ConfigPath())
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	rest := slices.Sorted(maps.Keys(cfg))
	rest = slices.DeleteFunc(rest, func(k string) bool { return slices.Contains(sectionOrder, k) })
	var sb strings.Builder
	sb.WriteString("# Ovi configuration\n# Edit directly or run: ovi --setup\n\n")
	for _, key := range append(slices.Clone(sectionOrder), rest...) {
		if _, ok := cfg[key]; !ok {
			continue
		}
		block, err := yaml.Marshal(map[string]any{key: cfg[key]})
		if err != nil {
			return err
		}
		sb.Write(block)
		sb.WriteString("\n")
	}
	return os.WriteFile(path, []byte(sb.String()), 0o600)
}

// RawDevices reads the device entries of a raw config (string or list).
func RawDevices(cfg map[string]any) []string {
	switch v := cfg["devices"].(type) {
	case string:
		return splitList(v)
	case []any:
		var out []string
		for _, item := range v {
			if s := strings.TrimSpace(fmt.Sprint(item)); s != "" {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

// splitList splits a comma-separated list, dropping blanks.
func splitList(s string) []string {
	var out []string
	for item := range strings.SplitSeq(s, ",") {
		if item = strings.TrimSpace(item); item != "" {
			out = append(out, item)
		}
	}
	return out
}

// SecretsPath is the ESPHome secrets file.
var SecretsPath = filepath.Join("esphome", "secrets.yaml")

// readEncryptionKey reads api_encryption_key from the secrets file.
func readEncryptionKey() string {
	data, err := os.ReadFile(SecretsPath)
	if err != nil {
		return ""
	}
	for line := range strings.SplitSeq(string(data), "\n") {
		if value, ok := strings.CutPrefix(line, "api_encryption_key:"); ok {
			return strings.Trim(strings.TrimSpace(value), `"'`)
		}
	}
	return ""
}

// AddDevicesToConfig adds host[:port[:key]] entries to the config file,
// leaving other settings alone. Bare hosts get the encryption key from
// the secrets file.
func AddDevicesToConfig(entries []string, path string) error {
	cfg, err := LoadConfigFile(path)
	if err != nil {
		return err
	}
	devices := RawDevices(cfg)
	known := map[string]bool{}
	for _, d := range devices {
		known[hostOf(d)] = true
	}
	key := readEncryptionKey()
	for _, entry := range entries {
		if !strings.Contains(entry, ":") && key != "" {
			entry += ":6055:" + key
		}
		if host := hostOf(entry); !known[host] {
			known[host] = true
			devices = append(devices, entry)
		}
	}
	cfg["devices"] = anyList(devices)
	return saveConfigFile(cfg, path)
}

func hostOf(entry string) string {
	host, _, _ := strings.Cut(entry, ":")
	return host
}

func anyList(items []string) []any {
	out := make([]any, len(items))
	for i, s := range items {
		out[i] = s
	}
	return out
}

// NeedsSetup reports whether there is no config file yet.
func NeedsSetup() bool {
	_, err := os.Stat(config.ConfigPath())
	return os.IsNotExist(err)
}
