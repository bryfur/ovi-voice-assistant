package main

import (
	"cmp"
	"context"
	"errors"
	"log/slog"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/config"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
	"github.com/bryfur/ovi-voice-assistant/internal/music/browser"
	"github.com/bryfur/ovi-voice-assistant/internal/pipeline"
)

// serve loads the models, connects the devices and runs until SIGINT or
// SIGTERM.
func serve(s *config.Settings, deviceArgs []string) error {
	transports, err := transports(s, deviceArgs)
	if err != nil {
		return err
	}
	slog.Info("Starting Ovi", "transport", s.Transport.Type, "codec", s.Transport.Codec,
		"stt", s.STT.Provider+"/"+s.STT.Model, "tts", s.TTS.Provider+"/"+s.TTS.Model, "llm", s.LLM.Model)
	va, err := pipeline.New(s)
	if err != nil {
		return err
	}
	if err := va.Load(); err != nil {
		return err
	}
	ctx := context.Background()
	if err := va.Start(ctx); err != nil {
		return err
	}
	services, closeBrowsers := browser.Start(ctx, s.Music.Services)
	defer closeBrowsers()
	player := music.NewPlayer(services)

	devices, err := pipeline.NewManager(transports, s.Transport.Codec, va, player)
	if err != nil {
		return err
	}
	sched := scheduler.New(config.ExpandUser(s.Automations.Path),
		func(ctx context.Context, prompt string) (string, error) { return va.Agent.Ask(ctx, prompt, nil) },
		devices.Announce)
	devices.SetScheduler(sched)
	if err := devices.Start(); err != nil {
		return err
	}
	sched.Start()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	slog.Info("Shutting down...")
	sched.Stop()
	player.Stop()
	devices.Stop()
	va.Stop()
	return nil
}

// transports picks the devices to serve: the BLE device from the config,
// or the WiFi devices named on the command line, else those configured.
func transports(s *config.Settings, args []string) ([]device.Transport, error) {
	if s.Transport.Type == "ble" {
		t, err := device.NewBLE(s.BLE.DeviceName, s.BLE.DeviceAddress)
		if err != nil {
			return nil, err
		}
		return []device.Transport{t}, nil
	}
	devices, err := config.ParseDevices(cmp.Or(strings.Join(args, ","), string(s.Devices)))
	if err != nil {
		return nil, err
	}
	if len(devices) == 0 {
		return nil, errors.New("no devices configured; pass a host, run `ovi --scan`, or set devices in ~/.ovi/config.yaml")
	}
	var out []device.Transport
	for _, d := range devices {
		slog.Info("Device", "host", d.Host, "port", d.Port)
		if d.EncryptionKey != "" {
			slog.Warn("Encryption keys are not used yet; the connection is unencrypted", "host", d.Host)
		}
		out = append(out, device.NewWiFi(d.Host, d.Port))
	}
	return out, nil
}
