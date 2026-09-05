package config

import (
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoadRaw reads the YAML config as a generic map (empty if missing).
func LoadRaw(path string) (map[string]any, error) {
	if path == "" {
		path = ConfigPath()
	}
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return map[string]any{}, nil
		}
		return nil, err
	}
	out := map[string]any{}
	if err := yaml.Unmarshal(data, &out); err != nil {
		return nil, err
	}
	if out == nil {
		out = map[string]any{}
	}
	return out, nil
}

// SaveRaw writes a nested config map as YAML, one top-level section per block.
func SaveRaw(cfg map[string]any, path string) error {
	if path == "" {
		path = ConfigPath()
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	var sb strings.Builder
	sb.WriteString("# Ovi configuration\n# Edit directly or run: ovi --setup\n\n")
	keys := sectionOrder(cfg)
	for _, key := range keys {
		block, err := yaml.Marshal(map[string]any{key: cfg[key]})
		if err != nil {
			return err
		}
		sb.Write(block)
		sb.WriteString("\n")
	}
	return os.WriteFile(path, []byte(sb.String()), 0o600)
}

// sectionOrder returns keys in a stable, human-friendly order.
func sectionOrder(cfg map[string]any) []string {
	preferred := []string{"llm", "stt", "tts", "devices", "transport", "mic", "ble", "memory", "automations"}
	seen := map[string]bool{}
	var keys []string
	for _, k := range preferred {
		if _, ok := cfg[k]; ok {
			keys = append(keys, k)
			seen[k] = true
		}
	}
	var rest []string
	for k := range cfg {
		if !seen[k] {
			rest = append(rest, k)
		}
	}
	sortStrings(rest)
	return append(keys, rest...)
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// RawDevices extracts the device list from a raw config (string or list).
func RawDevices(cfg map[string]any) []string {
	switch v := cfg["devices"].(type) {
	case string:
		return splitDevices(v)
	case []any:
		var out []string
		for _, item := range v {
			if s := strings.TrimSpace(strings.TrimSpace(anyString(item))); s != "" {
				out = append(out, s)
			}
		}
		return out
	}
	return nil
}

func anyString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case nil:
		return ""
	}
	b, _ := yaml.Marshal(v)
	return strings.TrimSpace(string(b))
}

func splitDevices(s string) []string {
	var out []string
	for _, d := range strings.Split(s, ",") {
		if d = strings.TrimSpace(d); d != "" {
			out = append(out, d)
		}
	}
	return out
}

// SecretsPath is the ESPHome secrets file.
var SecretsPath = filepath.Join("esphome", "secrets.yaml")

// ReadEncryptionKey reads api_encryption_key from esphome/secrets.yaml.
func ReadEncryptionKey() string {
	data, err := os.ReadFile(SecretsPath)
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "api_encryption_key:") {
			val := strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
			return strings.Trim(val, `"'`)
		}
	}
	return ""
}

// AddDevicesToConfig adds device entries to the config file without
// touching other settings. Each entry is host[:port[:key]]; bare hostnames
// get the encryption key from esphome/secrets.yaml automatically.
func AddDevicesToConfig(entries []string, path string) error {
	if path == "" {
		path = ConfigPath()
	}
	existing, err := LoadRaw(path)
	if err != nil {
		return err
	}
	current := RawDevices(existing)
	key := ReadEncryptionKey()
	hosts := map[string]bool{}
	for _, e := range current {
		hosts[strings.SplitN(e, ":", 2)[0]] = true
	}
	for _, entry := range entries {
		if !strings.Contains(entry, ":") && key != "" {
			entry = entry + ":6055:" + key
		}
		host := strings.SplitN(entry, ":", 2)[0]
		if !hosts[host] {
			current = append(current, entry)
			hosts[host] = true
		}
	}
	list := make([]any, len(current))
	for i, d := range current {
		list[i] = d
	}
	existing["devices"] = list
	return SaveRaw(existing, path)
}

// NeedsSetup reports whether first-run setup is needed (no config file).
func NeedsSetup() bool {
	_, err := os.Stat(ConfigPath())
	return os.IsNotExist(err)
}
