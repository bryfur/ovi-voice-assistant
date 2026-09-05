// Package config holds Ovi's layered settings: built-in defaults, then
// ~/.ovi/config.yaml, then a .env file, then OVI_ environment variables
// (OVI_SECTION__KEY), then command-line overrides.
package config

import (
	"cmp"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/joho/godotenv"
	"gopkg.in/yaml.v3"
)

const (
	envPrefix  = "OVI_"
	devicePort = 6055 // the ESPHome component's TCP port
)

func home() string {
	h, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return h
}

// ConfigPath is ~/.ovi/config.yaml.
func ConfigPath() string { return filepath.Join(home(), ".ovi", "config.yaml") }

// CacheDir is ~/.cache/ovi.
func CacheDir() string { return filepath.Join(home(), ".cache", "ovi") }

// ExpandUser replaces a leading "~" with the home directory.
func ExpandUser(p string) string {
	if rest, ok := strings.CutPrefix(p, "~"); ok && (rest == "" || rest[0] == '/') {
		return filepath.Join(home(), rest)
	}
	return p
}

// DeviceConfig addresses one ESPHome voice device.
type DeviceConfig struct {
	Host          string
	Port          int
	EncryptionKey string
}

// ParseDevices reads a comma-separated list of host[:port[:key]].
func ParseDevices(raw string) ([]DeviceConfig, error) {
	var devices []DeviceConfig
	for entry := range strings.SplitSeq(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		host, rest, _ := strings.Cut(entry, ":")
		port, key, _ := strings.Cut(rest, ":")
		d := DeviceConfig{Host: host, Port: devicePort, EncryptionKey: key}
		if port != "" {
			var err error
			if d.Port, err = strconv.Atoi(port); err != nil {
				return nil, fmt.Errorf("invalid port in device %q: %w", entry, err)
			}
		}
		devices = append(devices, d)
	}
	return devices, nil
}

// LLMConfig configures the language model endpoint and the agent.
type LLMConfig struct {
	APIKey       string `yaml:"api_key"`
	BaseURL      string `yaml:"base_url"`
	Model        string `yaml:"model"`
	Instructions string `yaml:"instructions"`
	MCPServers   string `yaml:"mcp_servers"` // JSON array, or @path to one
	Agents       string `yaml:"agents"`      // JSON array of sub-agents, or @path
	// Reasoning false asks thinking models not to think before answering
	// (reasoning_effort=none, plus enable_thinking=false / think=false for
	// local servers). Leave true to send nothing.
	Reasoning bool `yaml:"reasoning"`
}

// STTConfig configures speech-to-text.
type STTConfig struct {
	Provider string  `yaml:"provider"` // nemotron | whisper
	Model    string  `yaml:"model"`    // nemotron: chunk (80ms|160ms|560ms|1120ms); whisper: tiny.en, base.en, ...
	Language string  `yaml:"language"` // whisper multilingual models only
	Silence  float64 `yaml:"silence"`  // seconds of silence that end an utterance; the main latency knob
}

// TTSConfig configures text-to-speech.
type TTSConfig struct {
	Provider string  `yaml:"provider"` // kokoro | piper
	Model    string  `yaml:"model"`    // kokoro voice (af_heart) or piper voice (en_US-lessac-medium)
	Speed    float64 `yaml:"speed"`    // 1.0 = normal
}

// TransportConfig configures the device link and codec.
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

// AutomationsConfig says where automations are stored.
type AutomationsConfig struct {
	Path string `yaml:"path"`
}

// DeviceList is a comma-separated device string that also accepts a YAML
// sequence.
type DeviceList string

func (d *DeviceList) UnmarshalYAML(node *yaml.Node) error {
	switch node.Kind {
	case yaml.SequenceNode:
		items := make([]string, len(node.Content))
		for i, n := range node.Content {
			items[i] = n.Value
		}
		*d = DeviceList(strings.Join(items, ","))
	case yaml.ScalarNode:
		*d = DeviceList(node.Value)
	default:
		return fmt.Errorf("devices: expected string or list, got %v", node.Kind)
	}
	return nil
}

// Settings is the complete Ovi configuration.
type Settings struct {
	LLM         LLMConfig         `yaml:"llm"`
	STT         STTConfig         `yaml:"stt"`
	TTS         TTSConfig         `yaml:"tts"`
	Transport   TransportConfig   `yaml:"transport"`
	BLE         BLEConfig         `yaml:"ble"`
	Music       MusicConfig       `yaml:"music"`
	Automations AutomationsConfig `yaml:"automations"`
	Devices     DeviceList        `yaml:"devices"` // host[:port[:key]], comma-separated or a list
}

// DefaultInstructions is the system prompt of the voice agent.
const DefaultInstructions = "You are a voice assistant. Your responses will be spoken aloud. " +
	"Rules: Reply in 8-9 short sentences if possible. Never explain your reasoning. " +
	"Never use markdown, bullet points, or lists. Never include internal thoughts. " +
	"Just give a direct, natural spoken answer. " +
	"If your response asks a question or requires a follow-up from the user, " +
	"end your response with [LISTEN]. Only use [LISTEN] when you need the user to respond."

// Default returns the built-in defaults.
func Default() *Settings {
	return &Settings{
		LLM:         LLMConfig{Model: "gpt-4o-mini", Instructions: DefaultInstructions, Reasoning: true},
		STT:         STTConfig{Provider: "nemotron", Model: "560ms", Language: "en", Silence: 0.75},
		TTS:         TTSConfig{Provider: "kokoro", Model: "af_heart", Speed: 1},
		Transport:   TransportConfig{Type: "wifi", Codec: "lc3"},
		Automations: AutomationsConfig{Path: "~/.ovi/automations.json"},
	}
}

// LoadOptions says where Load reads from; zero values mean the defaults.
type LoadOptions struct {
	ConfigPath  string            // YAML file; empty means ConfigPath()
	EnvFile     string            // dotenv file; empty means ".env"
	SkipEnvFile bool              // do not read a dotenv file
	Environ     []string          // process environment; nil means os.Environ()
	Overrides   map[string]string // dotted keys ("stt.provider", "devices") applied last
}

// Load builds Settings from every layer.
func Load(opts LoadOptions) (*Settings, error) {
	s := Default()
	path := cmp.Or(opts.ConfigPath, ConfigPath())
	data, err := os.ReadFile(path)
	if err == nil {
		err = yaml.Unmarshal(data, s)
	} else if os.IsNotExist(err) {
		err = nil
	}
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}

	env := map[string]string{}
	if !opts.SkipEnvFile {
		fromFile, _ := godotenv.Read(cmp.Or(opts.EnvFile, ".env"))
		for k, v := range fromFile {
			env[k] = v
		}
	}
	environ := opts.Environ
	if environ == nil {
		environ = os.Environ()
	}
	for _, kv := range environ {
		if k, v, ok := strings.Cut(kv, "="); ok {
			env[k] = v
		}
	}
	if err := s.applyEnv(env); err != nil {
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
	v := reflect.ValueOf(s).Elem()
	parts := strings.Split(key, ".")
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

// applyEnv assigns OVI_<SECTION>__<KEY> and OVI_<KEY> variables.
func (s *Settings) applyEnv(env map[string]string) error {
	v := reflect.ValueOf(s).Elem()
	for i := range v.NumField() {
		tag := strings.ToUpper(v.Type().Field(i).Tag.Get("yaml"))
		f := v.Field(i)
		if f.Kind() != reflect.Struct {
			if val, ok := env[envPrefix+tag]; ok {
				if err := setField(f, val); err != nil {
					return fmt.Errorf("%s: %w", envPrefix+tag, err)
				}
			}
			continue
		}
		for j := range f.NumField() {
			name := envPrefix + tag + "__" + strings.ToUpper(f.Type().Field(j).Tag.Get("yaml"))
			if val, ok := env[name]; ok {
				if err := setField(f.Field(j), val); err != nil {
					return fmt.Errorf("%s: %w", name, err)
				}
			}
		}
	}
	return nil
}

func fieldByTag(v reflect.Value, tag string) (reflect.Value, bool) {
	for i := range v.NumField() {
		if v.Type().Field(i).Tag.Get("yaml") == tag {
			return v.Field(i), true
		}
	}
	return reflect.Value{}, false
}

func setField(f reflect.Value, value string) error {
	value = strings.TrimSpace(value)
	switch f.Kind() {
	case reflect.String:
		f.SetString(value)
	case reflect.Int:
		n, err := strconv.Atoi(value)
		if err != nil {
			return err
		}
		f.SetInt(int64(n))
	case reflect.Float64:
		n, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return err
		}
		f.SetFloat(n)
	case reflect.Bool:
		switch strings.ToLower(value) {
		case "1", "true", "yes", "on", "y", "t":
			f.SetBool(true)
		case "0", "false", "no", "off", "n", "f", "":
			f.SetBool(false)
		default:
			return fmt.Errorf("invalid boolean %q", value)
		}
	case reflect.Slice:
		var items []string
		for item := range strings.SplitSeq(value, ",") {
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
