package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"github.com/bryfur/ovi-voice-assistant/internal/codec"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/memory"
	"github.com/bryfur/ovi-voice-assistant/internal/pipeline"
	"github.com/bryfur/ovi-voice-assistant/internal/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/transport"
)

// serve loads models and runs the server until SIGINT/SIGTERM.
func serve(settings *config.Settings, deviceArgs []string) error {
	// Determine the TTS sample rate. First create a probe codec to find the
	// actual rate it'll use (e.g., LC3 snaps 22050→24000), then create TTS
	// at that rate.
	probeRate := settings.Transport.SpeakerSampleRate
	if probeRate == 0 {
		probeRate = 24000
	}
	probe, err := codec.CreateNamed(settings.Transport.Codec, probeRate, 1, 0)
	if err != nil {
		return err
	}
	ttsRate := probe.SampleRate()

	slog.Info("Starting Ovi")
	slog.Info("  Transport", "type", settings.Transport.Type)
	slog.Info("  Codec", "name", settings.Transport.Codec)
	slog.Info("  STT", "provider", settings.STT.Provider, "model", settings.STT.Model)
	slog.Info("  TTS", "provider", settings.TTS.Provider, "model", settings.TTS.Model)
	slog.Info("  Agent", "model", settings.LLM.Model)
	if settings.LLM.MCPServers != "" {
		slog.Info("  MCP", "servers", settings.LLM.MCPServers)
	}

	// Resolve devices before loading models so misconfiguration fails fast.
	var devices []config.DeviceConfig
	if settings.Transport.Type != "ble" {
		devices, err = resolveDevices(deviceArgs, settings)
		if err != nil {
			return err
		}
	}

	// Initialize memory if enabled
	var mem *memory.Memory
	if settings.Memory.Enabled {
		mem = memory.New(settings)
		if err := mem.Load(); err != nil {
			return fmt.Errorf("load memory: %w", err)
		}
		slog.Info("  Memory: SQLite", "bank", settings.Memory.BankID)
	}

	// Load models — TTS uses the codec's actual sample rate
	pl, err := pipeline.New(settings, ttsRate)
	if err != nil {
		return err
	}
	if err := pl.Load(); err != nil {
		return err
	}
	actualRate := pl.TTS.SampleRate()
	slog.Info("  Speaker sample rate", "hz", actualRate)

	if settings.Transport.Type == "ble" {
		return serveBLE(settings, pl, actualRate, mem)
	}
	return serveWiFi(devices, settings, pl, actualRate, mem)
}

func resolveDevices(cliDevices []string, settings *config.Settings) ([]config.DeviceConfig, error) {
	var devices []config.DeviceConfig
	var err error
	if len(cliDevices) > 0 {
		joined := ""
		for i, d := range cliDevices {
			if i > 0 {
				joined += ","
			}
			joined += d
		}
		devices, err = config.ParseDevices(joined)
	} else {
		devices, err = settings.GetDevices()
	}
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return nil, errors.New("No devices configured. Usage:\n" +
			"  ovi voice-pe-XXXX.local           # connect by mDNS hostname\n" +
			"  ovi 192.168.1.100                  # connect by IP\n" +
			"  ovi --scan                         # discover devices on network\n" +
			"  ovi 192.168.1.100:6055:KEY          # with encryption key\n" +
			"  ovi --transport ble                 # connect over Bluetooth\n")
	}
	anyEncrypted := false
	for i, d := range devices {
		slog.Info("  Device", "index", i, "host", d.Host, "port", d.Port)
		if d.EncryptionKey != "" {
			anyEncrypted = true
		}
	}
	if !anyEncrypted {
		slog.Warn("No encryption configured. Communication is unencrypted. " +
			"Run 'ovi --gen-key' to generate a key and enable encryption.")
	}
	return devices, nil
}

func createScheduler(settings *config.Settings, pl *pipeline.VoiceAssistant, announce scheduler.Announce) *scheduler.Scheduler {
	s := scheduler.New(config.ExpandUser(settings.Automations.Path),
		func(ctx context.Context, prompt string) (string, error) {
			return pl.Agent.RunText(ctx, prompt, nil)
		}, announce)
	s.Load()
	return s
}

func waitForSignal() {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	signal.Stop(sig)
}

func shutdown(sched *scheduler.Scheduler, stopDevices func(ctx context.Context), pl *pipeline.VoiceAssistant, mem *memory.Memory) {
	slog.Info("Shutting down...")
	sched.Stop()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	stopDevices(ctx)
	cancel()
	ctx, cancel = context.WithTimeout(context.Background(), 5*time.Second)
	_ = pl.Stop(ctx)
	cancel()
	if mem != nil {
		mem.Close()
	}
}

// serveWiFi serves WiFi devices via DeviceManager.
func serveWiFi(devices []config.DeviceConfig, settings *config.Settings, pl *pipeline.VoiceAssistant, ttsRate int, mem *memory.Memory) error {
	ctx := context.Background()
	if err := pl.Start(ctx); err != nil {
		return err
	}
	mgr, err := device.NewDeviceManager(devices, settings, pl, ttsRate, nil)
	if err != nil {
		return err
	}
	if err := mgr.Start(); err != nil {
		return err
	}
	sched := createScheduler(settings, pl, mgr.AnnounceAll)
	mgr.SetScheduler(sched)
	if mem != nil {
		mgr.SetMemory(mem)
	}
	sched.Start()

	waitForSignal()
	shutdown(sched, mgr.Stop, pl, mem)
	return nil
}

// serveBLE serves a single BLE device.
func serveBLE(settings *config.Settings, pl *pipeline.VoiceAssistant, ttsRate int, mem *memory.Memory) error {
	ctx := context.Background()
	t, err := transport.NewBLETransport(settings.BLE.DeviceName, settings.BLE.DeviceAddress)
	if err != nil {
		return err
	}
	c, err := codec.CreateNamed(settings.Transport.Codec, ttsRate, 1, 0)
	if err != nil {
		return err
	}
	if err := pl.Start(ctx); err != nil {
		return err
	}
	conn := device.NewDeviceConnection(t, c, pl, settings, device.Options{})
	if err := conn.Start(); err != nil {
		return err
	}
	sched := createScheduler(settings, pl, func(ctx context.Context, text string) error {
		conn.Announce(text)
		return nil
	})
	conn.SetScheduler(sched)
	if mem != nil {
		conn.SetMemory(agent.MemoryStore(mem))
	}
	sched.Start()

	waitForSignal()
	shutdown(sched, func(context.Context) { _ = conn.Stop() }, pl, mem)
	return nil
}
