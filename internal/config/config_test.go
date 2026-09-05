package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseDevicesHostOnly(t *testing.T) {
	devs, err := ParseDevices("192.168.1.42")

	if err != nil || len(devs) != 1 {
		t.Fatalf("got %v, %v", devs, err)
	}
	if devs[0].Host != "192.168.1.42" || devs[0].Port != 6055 || devs[0].EncryptionKey != "" {
		t.Fatalf("got %+v", devs[0])
	}
}

func TestParseDevicesHostPortKey(t *testing.T) {
	devs, err := ParseDevices("host:9999:secret")

	if err != nil {
		t.Fatal(err)
	}
	if devs[0].Port != 9999 || devs[0].EncryptionKey != "secret" {
		t.Fatalf("got %+v", devs[0])
	}
}

func TestParseDevicesEmptyFieldsUseDefaults(t *testing.T) {
	devs, err := ParseDevices("host::")

	if err != nil {
		t.Fatal(err)
	}
	if devs[0].Port != 6055 || devs[0].EncryptionKey != "" {
		t.Fatalf("got %+v", devs[0])
	}
}

func TestParseDevicesMultipleWithWhitespaceAndTrailingComma(t *testing.T) {
	devs, err := ParseDevices(" a , b:1 ,c:2:k, ")

	if err != nil || len(devs) != 3 {
		t.Fatalf("got %v, %v", devs, err)
	}
	if devs[0].Host != "a" || devs[1].Port != 1 || devs[2].EncryptionKey != "k" {
		t.Fatalf("got %+v", devs)
	}
}

func TestParseDevicesEmptyInputs(t *testing.T) {
	for _, in := range []string{"", "   ", ",,,"} {
		devs, err := ParseDevices(in)

		if err != nil || len(devs) != 0 {
			t.Fatalf("ParseDevices(%q) = %v, %v", in, devs, err)
		}
	}
}

func TestParseDevicesNonIntegerPort(t *testing.T) {
	_, err := ParseDevices("host:abc")

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestDefaults(t *testing.T) {
	s := Default()

	if s.LLM.Model != "gpt-4o-mini" || s.STT.Provider != "nemotron" || s.TTS.Provider != "kokoro" {
		t.Fatalf("unexpected defaults: %+v", s)
	}
	if s.Transport.Codec != "lc3" || s.Mic.SampleRate != 16000 || !s.Memory.Enabled {
		t.Fatalf("unexpected defaults: %+v", s)
	}
}

func TestGetDevices(t *testing.T) {
	s := Default()
	s.Devices = "a,b:1"

	devs, err := s.GetDevices()

	if err != nil || len(devs) != 2 || devs[1].Port != 1 {
		t.Fatalf("got %v, %v", devs, err)
	}
}

func TestLoadYamlValuesApplied(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	os.WriteFile(path, []byte("llm:\n  model: llama3\nstt:\n  provider: whisper\ndevices:\n  - a.local\n  - b.local\n"), 0o644)

	s, err := Load(LoadOptions{ConfigPath: path, SkipEnvFile: true, Environ: []string{}})

	if err != nil {
		t.Fatal(err)
	}
	if s.LLM.Model != "llama3" || s.STT.Provider != "whisper" || s.Devices != "a.local,b.local" {
		t.Fatalf("got %+v", s)
	}
	if s.TTS.Provider != "kokoro" {
		t.Fatal("defaults should survive partial YAML")
	}
}

func TestLoadEnvOverridesYaml(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	os.WriteFile(path, []byte("llm:\n  model: llama3\n"), 0o644)

	s, err := Load(LoadOptions{
		ConfigPath:  path,
		SkipEnvFile: true,
		Environ:     []string{"OVI_LLM__MODEL=gpt-4o", "OVI_DEVICES=x.local", "OVI_MEMORY__ENABLED=false", "OVI_MIC__SAMPLE_RATE=8000"},
	})

	if err != nil {
		t.Fatal(err)
	}
	if s.LLM.Model != "gpt-4o" || s.Devices != "x.local" || s.Memory.Enabled || s.Mic.SampleRate != 8000 {
		t.Fatalf("got %+v", s)
	}
}

func TestLoadOverridesBeatEnv(t *testing.T) {
	s, err := Load(LoadOptions{
		ConfigPath:  filepath.Join(t.TempDir(), "missing.yaml"),
		SkipEnvFile: true,
		Environ:     []string{"OVI_LLM__MODEL=gpt-4o"},
		Overrides:   map[string]string{"llm.model": "cli-model", "transport.codec": "opus"},
	})

	if err != nil {
		t.Fatal(err)
	}
	if s.LLM.Model != "cli-model" || s.Transport.Codec != "opus" {
		t.Fatalf("got %+v", s)
	}
}

func TestLoadDotenvBelowEnv(t *testing.T) {
	dir := t.TempDir()
	envFile := filepath.Join(dir, ".env")
	os.WriteFile(envFile, []byte("OVI_LLM__MODEL=dotenv-model\nOVI_TTS__MODEL=dotenv-voice\n"), 0o644)

	s, err := Load(LoadOptions{
		ConfigPath: filepath.Join(dir, "missing.yaml"),
		EnvFile:    envFile,
		Environ:    []string{"OVI_LLM__MODEL=real-env"},
	})

	if err != nil {
		t.Fatal(err)
	}
	if s.LLM.Model != "real-env" || s.TTS.Model != "dotenv-voice" {
		t.Fatalf("got %+v", s)
	}
}

func TestSetUnknownKey(t *testing.T) {
	s := Default()

	err := s.Set("nope.key", "x")

	if err == nil {
		t.Fatal("expected error")
	}
}

func TestSetPointerInt(t *testing.T) {
	s := Default()

	err := s.Set("tts.speaker_id", "3")

	if err != nil || s.TTS.SpeakerID == nil || *s.TTS.SpeakerID != 3 {
		t.Fatalf("got %v, %v", s.TTS.SpeakerID, err)
	}
}

func TestExpandUser(t *testing.T) {
	home, _ := os.UserHomeDir()

	got := ExpandUser("~/.ovi/x")

	if got != filepath.Join(home, ".ovi/x") {
		t.Fatalf("got %q", got)
	}
}
