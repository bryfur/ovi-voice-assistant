// Package config holds the layered settings for Ovi.
//
// Sources are applied in this order (later overrides earlier):
//
//  1. Built-in defaults
//  2. YAML config file (~/.ovi/config.yaml)
//  3. .env file in the working directory
//  4. Environment variables (OVI_ prefix, "__" as the nesting delimiter)
//  5. CLI overrides
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

// EnvPrefix is the prefix for all environment variable overrides.
const EnvPrefix = "OVI_"

// DefaultDevicePort is the TCP port the ESPHome component listens on.
const DefaultDevicePort = 6055

// ConfigDir returns ~/.ovi.
func ConfigDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".ovi")
}

// ConfigPath returns ~/.ovi/config.yaml.
func ConfigPath() string {
	return filepath.Join(ConfigDir(), "config.yaml")
}

// CacheDir returns ~/.cache/ovi.
func CacheDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".cache", "ovi")
}

// ExpandUser replaces a leading "~" with the user's home directory.
func ExpandUser(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") {
		home, err := os.UserHomeDir()
		if err == nil {
			return filepath.Join(home, strings.TrimPrefix(p, "~"))
		}
	}
	return p
}

// DeviceConfig describes a single ESPHome voice device.
type DeviceConfig struct {
	Host          string
	Port          int
	EncryptionKey string
}

// ParseDevices parses comma-separated device strings.
//
// Format: host[:port[:key]]
func ParseDevices(raw string) ([]DeviceConfig, error) {
	var devices []DeviceConfig
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		parts := strings.SplitN(entry, ":", 3)
		dev := DeviceConfig{Host: parts[0], Port: DefaultDevicePort}
		if len(parts) > 1 && parts[1] != "" {
			port, err := strconv.Atoi(parts[1])
			if err != nil {
				return nil, fmt.Errorf("invalid port in device %q: %w", entry, err)
			}
			dev.Port = port
		}
		if len(parts) > 2 && parts[2] != "" {
			dev.EncryptionKey = parts[2]
		}
		devices = append(devices, dev)
	}
	return devices, nil
}

// ── Nested config sections ───────────────────────────────────

// LLMConfig configures the language model endpoint and agent.
type LLMConfig struct {
	APIKey       string `yaml:"api_key"`
	BaseURL      string `yaml:"base_url"`
	Model        string `yaml:"model"`
	Instructions string `yaml:"instructions"`
	MCPServers   string `yaml:"mcp_servers"`
	Agents       string `yaml:"agents"`
}

// STTConfig configures speech-to-text.
type STTConfig struct {
	Provider string `yaml:"provider"` // nemotron | whisper
	Model    string `yaml:"model"`    // nemotron: chunk (80ms|160ms|560ms|1120ms); whisper: tiny.en, base.en, ...
	Language string `yaml:"language"` // whisper multilingual models only
}

// TTSConfig configures text-to-speech.
type TTSConfig struct {
	Provider string  `yaml:"provider"` // kokoro | piper
	Model    string  `yaml:"model"`    // kokoro voice (af_heart) or piper voice (en_US-lessac-medium)
	Speed    float64 `yaml:"speed"`    // 1.0 = normal
}

// TransportConfig configures the device transport and codec.
type TransportConfig struct {
	Type  string `yaml:"type"`  // wifi | ble
	Codec string `yaml:"codec"` // pcm | lc3 | opus
}

// BLEConfig selects the BLE device.
type BLEConfig struct {
	DeviceName    string `yaml:"device_name"`
	DeviceAddress string `yaml:"device_address"`
}

// MusicConfig enables browser-based music services.
type MusicConfig struct {
	Services []string `yaml:"services"` // spotify, apple (YouTube Music is always available)
}

// AutomationsConfig configures the scheduler persistence path.
type AutomationsConfig struct {
	Path string `yaml:"path"`
}

// DeviceList is a comma-separated device string that also accepts a YAML
// sequence.
type DeviceList string

// UnmarshalYAML accepts either a scalar or a sequence of scalars.
func (d *DeviceList) UnmarshalYAML(node *yaml.Node) error {
	switch node.Kind {
	case yaml.SequenceNode:
		items := make([]string, 0, len(node.Content))
		for _, n := range node.Content {
			items = append(items, n.Value)
		}
		*d = DeviceList(strings.Join(items, ","))
		return nil
	case yaml.ScalarNode:
		*d = DeviceList(node.Value)
		return nil
	default:
		return fmt.Errorf("devices: expected string or list, got %v", node.Kind)
	}
}

// ── Main settings ────────────────────────────────────────────

// Settings is the complete Ovi configuration.
type Settings struct {
	LLM         LLMConfig         `yaml:"llm"`
	STT         STTConfig         `yaml:"stt"`
	TTS         TTSConfig         `yaml:"tts"`
	Transport   TransportConfig   `yaml:"transport"`
	BLE         BLEConfig         `yaml:"ble"`
	Music       MusicConfig       `yaml:"music"`
	Automations AutomationsConfig `yaml:"automations"`

	// Devices — comma-separated string or YAML list: host[:port[:key]]
	Devices DeviceList `yaml:"devices"`
}

// DefaultInstructions is the default system prompt for the voice agent.
const DefaultInstructions = "You are a voice assistant. Your responses will be spoken aloud. " +
	"Rules: Reply in 8-9 short sentences if possible. Never explain your reasoning. " +
	"Never use markdown, bullet points, or lists. Never include internal thoughts. " +
	"Just give a direct, natural spoken answer. " +
	"If your response asks a question or requires a follow-up from the user, " +
	"end your response with [LISTEN]. Only use [LISTEN] when you need the user to respond."

// Default returns the built-in defaults.
func Default() *Settings {
	return &Settings{
		LLM: LLMConfig{
			Model:        "gpt-4o-mini",
			Instructions: DefaultInstructions,
		},
		STT:         STTConfig{Provider: "nemotron", Model: "560ms", Language: "en"},
		TTS:         TTSConfig{Provider: "kokoro", Model: "af_heart", Speed: 1},
		Transport:   TransportConfig{Type: "wifi", Codec: "lc3"},
		Automations: AutomationsConfig{Path: "~/.ovi/automations.json"},
	}
}

// GetDevices parses the configured device list.
func (s *Settings) GetDevices() ([]DeviceConfig, error) {
	if s.Devices == "" {
		return nil, nil
	}
	return ParseDevices(string(s.Devices))
}

// LoadOptions controls where Load reads from.
type LoadOptions struct {
	// ConfigPath is the YAML file to read. Empty means ConfigPath().
	ConfigPath string
	// EnvFile is the dotenv file to read. Empty means ".env".
	EnvFile string
	// SkipEnvFile disables dotenv loading.
	SkipEnvFile bool
	// Environ is the process environment; nil means os.Environ().
	Environ []string
	// Overrides are dotted keys ("stt.provider", "devices") applied last.
	Overrides map[string]string
}

// Load builds Settings from all layered sources.
func Load(opts LoadOptions) (*Settings, error) {
	s := Default()

	path := opts.ConfigPath
	if path == "" {
		path = ConfigPath()
	}
	if data, err := os.ReadFile(path); err == nil {
		if err := yaml.Unmarshal(data, s); err != nil {
			return nil, fmt.Errorf("parse %s: %w", path, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}

	env := map[string]string{}
	if !opts.SkipEnvFile {
		envFile := opts.EnvFile
		if envFile == "" {
			envFile = ".env"
		}
		if m, err := godotenv.Read(envFile); err == nil {
			for k, v := range m {
				env[k] = v
			}
		}
	}
	environ := opts.Environ
	if environ == nil {
		environ = os.Environ()
	}
	for _, kv := range environ {
		k, v, ok := strings.Cut(kv, "=")
		if ok {
			env[k] = v
		}
	}
	if err := applyEnv(s, env); err != nil {
		return nil, err
	}
	for key, value := range opts.Overrides {
		if err := s.Set(key, value); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// Set assigns a dotted key such as "stt.provider" or "devices".
func (s *Settings) Set(key, value string) error {
	parts := strings.Split(key, ".")
	v := reflect.ValueOf(s).Elem()
	for i, part := range parts {
		f, ok := fieldByTag(v, part)
		if !ok {
			return fmt.Errorf("unknown config key %q", key)
		}
		if i == len(parts)-1 {
			return setField(f, value)
		}
		if f.Kind() != reflect.Struct {
			return fmt.Errorf("config key %q is not a section", strings.Join(parts[:i+1], "."))
		}
		v = f
	}
	return nil
}

func applyEnv(s *Settings, env map[string]string) error {
	v := reflect.ValueOf(s).Elem()
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("yaml")
		if tag == "" {
			continue
		}
		f := v.Field(i)
		if f.Kind() == reflect.Struct {
			st := f.Type()
			for j := 0; j < st.NumField(); j++ {
				subTag := st.Field(j).Tag.Get("yaml")
				if subTag == "" {
					continue
				}
				name := EnvPrefix + strings.ToUpper(tag) + "__" + strings.ToUpper(subTag)
				if val, ok := env[name]; ok {
					if err := setField(f.Field(j), val); err != nil {
						return fmt.Errorf("%s: %w", name, err)
					}
				}
			}
			continue
		}
		name := EnvPrefix + strings.ToUpper(tag)
		if val, ok := env[name]; ok {
			if err := setField(f, val); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
		}
	}
	return nil
}

func fieldByTag(v reflect.Value, tag string) (reflect.Value, bool) {
	t := v.Type()
	for i := 0; i < t.NumField(); i++ {
		if t.Field(i).Tag.Get("yaml") == tag {
			return v.Field(i), true
		}
	}
	return reflect.Value{}, false
}

func setField(f reflect.Value, value string) error {
	switch f.Kind() {
	case reflect.String:
		f.SetString(value)
	case reflect.Int:
		n, err := strconv.Atoi(strings.TrimSpace(value))
		if err != nil {
			return err
		}
		f.SetInt(int64(n))
	case reflect.Float64:
		n, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
		if err != nil {
			return err
		}
		f.SetFloat(n)
	case reflect.Bool:
		b, err := parseBool(value)
		if err != nil {
			return err
		}
		f.SetBool(b)
	case reflect.Slice:
		var items []string
		for _, item := range strings.Split(value, ",") {
			if item = strings.TrimSpace(item); item != "" {
				items = append(items, item)
			}
		}
		f.Set(reflect.ValueOf(items))
	default:
		return fmt.Errorf("unsupported field type %s", f.Type())
	}
	return nil
}

func parseBool(value string) (bool, error) {
	switch strings.ToLower(strings.TrimSpace(value)) {
	case "1", "true", "yes", "on", "y", "t":
		return true, nil
	case "0", "false", "no", "off", "n", "f", "":
		return false, nil
	}
	return false, fmt.Errorf("invalid boolean %q", value)
}
