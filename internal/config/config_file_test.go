package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestSaveRawAndLoadRaw(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	cfg := map[string]any{
		"llm":     map[string]any{"model": "gpt-4o"},
		"devices": []any{"a.local", "b.local"},
	}

	if err := SaveRaw(cfg, path); err != nil {
		t.Fatal(err)
	}
	back, err := LoadRaw(path)

	if err != nil {
		t.Fatal(err)
	}
	if back["llm"].(map[string]any)["model"] != "gpt-4o" || len(RawDevices(back)) != 2 {
		t.Fatalf("got %+v", back)
	}
	data, _ := os.ReadFile(path)
	if !strings.HasPrefix(string(data), "# Ovi configuration") {
		t.Fatalf("missing header: %s", data)
	}
}

func TestLoadRawMissingIsEmpty(t *testing.T) {
	cfg, err := LoadRaw(filepath.Join(t.TempDir(), "nope.yaml"))

	if err != nil || len(cfg) != 0 {
		t.Fatalf("got %v, %v", cfg, err)
	}
}

func TestRawDevicesString(t *testing.T) {
	got := RawDevices(map[string]any{"devices": " a, b ,,"})

	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Fatalf("got %v", got)
	}
}

func TestAddDevicesToConfigDedupsAndAttachesKey(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "config.yaml")
	SaveRaw(map[string]any{"devices": []any{"a.local:6055:old"}, "llm": map[string]any{"model": "m"}}, path)
	secrets := filepath.Join(dir, "secrets.yaml")
	os.WriteFile(secrets, []byte("wifi_ssid: \"x\"\napi_encryption_key: \"KEY==\"\n"), 0o644)
	oldSecrets := SecretsPath
	SecretsPath = secrets
	defer func() { SecretsPath = oldSecrets }()

	err := AddDevicesToConfig([]string{"a.local", "b.local", "c.local:1"}, path)

	if err != nil {
		t.Fatal(err)
	}
	cfg, _ := LoadRaw(path)
	devs := RawDevices(cfg)
	if len(devs) != 3 || devs[0] != "a.local:6055:old" || devs[1] != "b.local:6055:KEY==" || devs[2] != "c.local:1" {
		t.Fatalf("devices = %v", devs)
	}
	if cfg["llm"].(map[string]any)["model"] != "m" {
		t.Fatal("other settings must be preserved")
	}
}

func TestReadEncryptionKeyMissing(t *testing.T) {
	oldSecrets := SecretsPath
	SecretsPath = filepath.Join(t.TempDir(), "none.yaml")
	defer func() { SecretsPath = oldSecrets }()

	if ReadEncryptionKey() != "" {
		t.Fatal("expected empty key")
	}
}
