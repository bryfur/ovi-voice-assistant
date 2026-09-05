package main

import (
	"context"
	"errors"
	"github.com/bryfur/ovi-voice-assistant/internal/agent"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
	"github.com/bryfur/ovi-voice-assistant/internal/pipeline"
)

// serve loads models and runs until SIGINT/SIGTERM.
func serve(s *config.Settings, deviceArgs []string) error {
	ble := s.Transport.Type == "ble"
	var devices []config.DeviceConfig
	if !ble {
		var err error
		if devices, err = resolveDevices(deviceArgs, s); err != nil {
			return err
		}
	}
	// TTS renders at the rate the codec will actually use (LC3 snaps 22050→24000).
	probe, err := device.NewCodec(s.Transport.Codec, 24000, 1, 0)
	if err != nil {
		return err
	}
	slog.Info("Starting Ovi", "transport", s.Transport.Type, "codec", s.Transport.Codec,
		"stt", s.STT.Provider+"/"+s.STT.Model, "tts", s.TTS.Provider+"/"+s.TTS.Model, "llm", s.LLM.Model)

	pl, err := pipeline.New(s, probe.SampleRate())
	if err != nil {
		return err
	}
	if err := pl.Load(); err != nil {
		return err
	}
	ctx := context.Background()
	if err := pl.Start(ctx); err != nil {
		return err
	}
	stopMusic := music.StartServices(ctx, s.Music.Services, 48000)
	defer stopMusic()

	var announce agent.AnnounceFunc
	var stopDevices func(context.Context)
	if ble {
		t, err := device.NewBLETransport(s.BLE.DeviceName, s.BLE.DeviceAddress)
		if err != nil {
			return err
		}
		c, err := device.NewCodec(s.Transport.Codec, pl.TTS.SampleRate(), 1, 0)
		if err != nil {
			return err
		}
		conn := pipeline.NewDeviceConnection(t, c, pl, s, pipeline.DeviceOptions{})
		if err := conn.Start(); err != nil {
			return err
		}
		announce = func(_ context.Context, text string) error { conn.Announce(text); return nil }
		stopDevices = func(context.Context) { _ = conn.Stop() }
		defer attachScheduler(s, pl, announce, conn.SetScheduler)()
	} else {
		mgr, err := pipeline.NewDeviceManager(devices, s, pl, pl.TTS.SampleRate(), nil)
		if err != nil {
			return err
		}
		if err := mgr.Start(); err != nil {
			return err
		}
		stopDevices = mgr.Stop
		defer attachScheduler(s, pl, mgr.AnnounceAll, mgr.SetScheduler)()
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	slog.Info("Shutting down...")
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	stopDevices(ctx)
	return pl.Stop(ctx)
}

// attachScheduler loads and starts the automation scheduler; the returned
// func stops it.
func attachScheduler(s *config.Settings, pl *pipeline.VoiceAssistant, announce agent.AnnounceFunc, attach func(*agent.Scheduler)) func() {
	sched := agent.NewScheduler(config.ExpandUser(s.Automations.Path),
		func(ctx context.Context, prompt string) (string, error) { return pl.Agent.RunText(ctx, prompt, nil) },
		announce)
	sched.Load()
	attach(sched)
	sched.Start()
	return sched.Stop
}

func resolveDevices(cli []string, s *config.Settings) ([]config.DeviceConfig, error) {
	var devices []config.DeviceConfig
	var err error
	if len(cli) > 0 {
		devices, err = config.ParseDevices(strings.Join(cli, ","))
	} else {
		devices, err = s.GetDevices()
	}
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return nil, errors.New("no devices configured; pass a host, run `ovi --scan`, or set devices in ~/.ovi/config.yaml")
	}
	encrypted := false
	for _, d := range devices {
		slog.Info("Device", "host", d.Host, "port", d.Port)
		encrypted = encrypted || d.EncryptionKey != ""
	}
	if !encrypted {
		slog.Warn("No encryption configured; run 'ovi --gen-key' to enable it")
	}
	return devices, nil
}
