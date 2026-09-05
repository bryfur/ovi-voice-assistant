package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeYAML(t *testing.T, dir, name, first string) {
	t.Helper()
	os.WriteFile(filepath.Join(dir, name), []byte(first+"\nesphome:\n  name: "+strings.TrimSuffix(name, ".yaml")+"\n"), 0o644)
}

func TestFindDeviceConfigsOrdersAndDescribes(t *testing.T) {
	dir := t.TempDir()
	writeYAML(t, dir, "zzz-board.yaml", "# ESPHome config for Some Board")
	writeYAML(t, dir, "voice-pe.yaml", "# ESPHome config for Home Assistant Voice PE")
	writeYAML(t, dir, "voice-pe-ble.yaml", "# ESPHome — BLE transport for Home Assistant Voice PE")
	writeYAML(t, dir, "atom-echo.yaml", "# ESPHome config for M5Stack ATOM Echo")
	writeYAML(t, dir, "plain.yaml", "no marker")
	os.WriteFile(filepath.Join(dir, "secrets.yaml"), []byte("x"), 0o644)

	configs := findDeviceConfigs(dir)

	names := make([]string, len(configs))
	for i, c := range configs {
		names[i] = c.Name
	}
	if strings.Join(names, ",") != "voice-pe,voice-pe-ble,atom-echo,plain,zzz-board" {
		t.Fatalf("order = %v", names)
	}
	if configs[0].Description != "Home Assistant Voice PE" || configs[1].Description != "Home Assistant Voice PE (BLE transport)" || configs[3].Description != "plain" {
		t.Fatalf("descriptions = %+v", configs)
	}
}

func TestCheckSecrets(t *testing.T) {
	dir := t.TempDir()
	good := filepath.Join(dir, "good.yaml")
	os.WriteFile(good, []byte("wifi_ssid: \"Home\"\napi_encryption_key: \"k\"\n"), 0o644)
	placeholder := filepath.Join(dir, "ph.yaml")
	os.WriteFile(placeholder, []byte("wifi_ssid: \"my_wifi_ssid\"\napi_encryption_key: \"k\"\n"), 0o644)

	if !checkSecrets(good) || checkSecrets(placeholder) || checkSecrets(filepath.Join(dir, "missing")) {
		t.Fatal("checkSecrets wrong")
	}
}

func TestWriteSecretsPreservesKeyAndExtras(t *testing.T) {
	path := filepath.Join(t.TempDir(), "secrets.yaml")
	os.WriteFile(path, []byte("wifi_ssid: \"old\"\napi_encryption_key: \"KEEP==\"\nother: \"x\"\n"), 0o644)

	key, err := writeSecrets(path, "New Net", "pw")

	if err != nil || key != "KEEP==" {
		t.Fatalf("key=%q err=%v", key, err)
	}
	data, _ := os.ReadFile(path)
	s := string(data)
	if !strings.Contains(s, `wifi_ssid: "New Net"`) || !strings.Contains(s, `wifi_password: "pw"`) || !strings.Contains(s, `other: "x"`) {
		t.Fatalf("content = %s", s)
	}
}

func TestWriteSecretsGeneratesKey(t *testing.T) {
	path := filepath.Join(t.TempDir(), "secrets.yaml")

	key, err := writeSecrets(path, "Net", "pw")

	if err != nil || len(key) != 44 {
		t.Fatalf("key=%q err=%v", key, err)
	}
}

func TestGenerateKeyLength(t *testing.T) {
	if len(GenerateKey()) != 44 {
		t.Fatal("expected base64 of 32 bytes")
	}
}

func TestDeviceName(t *testing.T) {
	dir := t.TempDir()
	writeYAML(t, dir, "voice-pe.yaml", "# x")

	if esphomeDeviceName(filepath.Join(dir, "voice-pe.yaml")) != "voice-pe" || esphomeDeviceName(filepath.Join(dir, "nope")) != "" {
		t.Fatal("esphomeDeviceName wrong")
	}
}
