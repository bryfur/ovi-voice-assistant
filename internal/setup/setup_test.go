package setup

import (
	"bytes"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/console"
	"github.com/bryfur/ovi-voice-assistant/internal/discovery"
)

func scripted(lines ...string) (*console.IO, *bytes.Buffer) {
	out := &bytes.Buffer{}
	return &console.IO{In: strings.NewReader(strings.Join(lines, "\n") + "\n"), Out: out}, out
}

func stubScan(t *testing.T, devices []discovery.Device) {
	t.Helper()
	oldScan, oldFlash := Scan, Flash
	Scan = func() []discovery.Device { return devices }
	Flash = func(*console.IO) { t.Fatal("flash should not run") }
	t.Cleanup(func() { Scan, Flash = oldScan, oldFlash })
}

func TestRunFirstTimeWritesConfig(t *testing.T) {
	stubScan(t, []discovery.Device{{Name: "voice-pe-1", Host: "voice-pe-1.local", IP: "10.0.0.5", Port: 6055}})
	path := filepath.Join(t.TempDir(), "config.yaml")
	c, out := scripted(
		"sk-test", // api key
		"",        // base url
		"gpt-4o",  // model
		"whisper", // stt provider
		"",        // whisper model (default whisper-1)
		"",        // stt base url
		"cpu",     // device
		"kokoro",  // tts provider
		"2",       // kokoro voice af_bella
		"n",       // flash?
		"y",       // scan?
		"all",     // add devices
		"",        // codec default lc3
		"y",       // save
	)

	cfg, err := Run(c, path)

	if err != nil {
		t.Fatal(err)
	}
	s, err := config.Load(config.LoadOptions{ConfigPath: path, SkipEnvFile: true, Environ: []string{}})
	if err != nil {
		t.Fatal(err)
	}
	if s.LLM.APIKey != "sk-test" || s.LLM.Model != "gpt-4o" || s.STT.Provider != "whisper" || s.STT.Model != "whisper-1" {
		t.Fatalf("loaded = %+v", s)
	}
	if s.TTS.Model != "af_bella" || s.Devices != "voice-pe-1.local" || s.Transport.Codec != "lc3" {
		t.Fatalf("loaded = %+v", s)
	}
	if cfg["devices"] == nil || !strings.Contains(out.String(), "Configuration saved") || !strings.Contains(out.String(), "****") {
		t.Fatalf("output = %s", out.String())
	}
}

func TestRunEditKeepsExistingDefaultsAndCanSkipSave(t *testing.T) {
	stubScan(t, nil)
	path := filepath.Join(t.TempDir(), "config.yaml")
	config.SaveRaw(map[string]any{
		"llm":     map[string]any{"model": "llama3", "base_url": "http://x"},
		"stt":     map[string]any{"provider": "nemotron", "model": "fp16"},
		"tts":     map[string]any{"provider": "piper", "model": "en_US-amy-low"},
		"devices": []any{"a.local"},
	}, path)
	c, out := scripted("", "", "", "", "", "", "", "", "n", "n", "", "", "n")

	cfg, err := Run(c, path)

	if err != nil {
		t.Fatal(err)
	}
	llm := cfg["llm"].(map[string]any)
	if llm["model"] != "llama3" || llm["base_url"] != "http://x" || cfg["stt"].(map[string]any)["model"] != "fp16" {
		t.Fatalf("cfg = %+v", cfg)
	}
	if cfg["tts"].(map[string]any)["model"] != "en_US-amy-low" || len(cfg["devices"].([]any)) != 1 {
		t.Fatalf("cfg = %+v", cfg)
	}
	if !strings.Contains(out.String(), "Configuration not saved") || !strings.Contains(out.String(), "Edit configuration") {
		t.Fatalf("output = %s", out.String())
	}
}

func TestSelectScannedDevicesByNumberMergesExisting(t *testing.T) {
	stubScan(t, []discovery.Device{
		{Name: "a", Host: "a.local", IP: "1", Port: 6055},
		{Name: "b", Host: "b.local", IP: "2", Port: 6055},
	})
	c, _ := scripted("2")

	got := selectScannedDevices(c, []string{"a.local"})

	if strings.Join(got, ",") != "a.local,b.local" {
		t.Fatalf("got %v", got)
	}
}

func TestSelectScannedDevicesNone(t *testing.T) {
	stubScan(t, []discovery.Device{{Name: "a", Host: "a.local"}})
	c, _ := scripted("none")

	got := selectScannedDevices(c, []string{"x.local"})

	if len(got) != 1 || got[0] != "x.local" {
		t.Fatalf("got %v", got)
	}
}
