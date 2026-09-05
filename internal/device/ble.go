package device

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"tinygo.org/x/bluetooth"
)

// GATT service and characteristics of the Ovi firmware.
var (
	bleService = uuid("BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0")
	bleMic     = uuid("BA5E0002-FADA-4C14-A34C-1AE0F0A0A0A0") // notify: mic frames
	bleSpeaker = uuid("BA5E0003-FADA-4C14-A34C-1AE0F0A0A0A0") // write: speaker frames
	bleControl = uuid("BA5E0004-FADA-4C14-A34C-1AE0F0A0A0A0") // notify + write: events
)

const (
	bleMTU         = 512 // write-without-response payload
	bleScanTimeout = 10 * time.Second
	bleRetry       = 2 * time.Second
)

func uuid(s string) bluetooth.UUID {
	u, err := bluetooth.ParseUUID(strings.ToLower(s))
	if err != nil {
		panic(err)
	}
	return u
}

// BLE speaks the GATT protocol. Events and audio are the same bytes as
// over WiFi, without the length prefix: [type byte][payload] on the
// control characteristic, raw frames on the audio ones.
type BLE struct {
	name, address string // find the device by either
	adapter       *bluetooth.Adapter

	mu      sync.Mutex
	h       Handler
	found   *bluetooth.Address
	dev     *bluetooth.Device
	speaker *bluetooth.DeviceCharacteristic
	control *bluetooth.DeviceCharacteristic
	closing bool
	ctx     context.Context
	cancel  context.CancelFunc
}

// NewBLE addresses a device by advertised name or MAC address.
func NewBLE(name, address string) (*BLE, error) {
	if name == "" && address == "" {
		return nil, errors.New("ble: a device name or address is required")
	}
	return &BLE{name: name, address: address, adapter: bluetooth.DefaultAdapter}, nil
}

func (t *BLE) String() string { return "ble:" + cmpOr(t.address, t.name) }

func (t *BLE) Connect(h Handler) error {
	t.mu.Lock()
	t.h, t.closing = h, false
	t.ctx, t.cancel = context.WithCancel(context.Background())
	first := t.found == nil
	t.mu.Unlock()
	if first {
		if err := t.adapter.Enable(); err != nil {
			return fmt.Errorf("ble: enable adapter: %w", err)
		}
		t.adapter.SetConnectHandler(t.onLink)
	}
	return t.dial()
}

func (t *BLE) Disconnect() error {
	t.mu.Lock()
	t.closing = true
	t.cancel()
	dev := t.dev
	t.dev = nil
	t.mu.Unlock()
	if dev != nil {
		return dev.Disconnect()
	}
	return nil
}

func (t *BLE) SendEvent(e Event, payload []byte) error {
	t.mu.Lock()
	c := t.control
	t.mu.Unlock()
	if c == nil {
		slog.Warn("Not connected, dropping event", "to", t, "event", e)
		return nil
	}
	_, err := c.WriteWithoutResponse(append([]byte{byte(e)}, payload...))
	return err
}

// SendAudio writes a frame in MTU-sized chunks.
func (t *BLE) SendAudio(frame []byte) error {
	t.mu.Lock()
	c := t.speaker
	t.mu.Unlock()
	if c == nil {
		slog.Warn("Not connected, dropping audio", "to", t)
		return nil
	}
	for len(frame) > 0 {
		n := min(len(frame), bleMTU)
		if _, err := c.WriteWithoutResponse(frame[:n]); err != nil {
			return err
		}
		frame = frame[n:]
	}
	return nil
}

// dial scans for the device (its address may change between sessions),
// connects and subscribes to its notifications.
func (t *BLE) dial() error {
	addr, err := t.scan()
	if err != nil {
		return err
	}
	dev, err := t.adapter.Connect(addr, bluetooth.ConnectionParams{})
	if err != nil {
		return fmt.Errorf("ble: connect: %w", err)
	}
	services, err := dev.DiscoverServices([]bluetooth.UUID{bleService})
	if err != nil || len(services) == 0 {
		_ = dev.Disconnect()
		return fmt.Errorf("ble: ovi service not found: %w", err)
	}
	chars, err := services[0].DiscoverCharacteristics([]bluetooth.UUID{bleMic, bleSpeaker, bleControl})
	if err != nil || len(chars) != 3 {
		_ = dev.Disconnect()
		return fmt.Errorf("ble: ovi characteristics not found: %w", err)
	}
	byUUID := map[bluetooth.UUID]*bluetooth.DeviceCharacteristic{}
	for i := range chars {
		byUUID[chars[i].UUID()] = &chars[i]
	}
	t.mu.Lock()
	t.found, t.dev, t.speaker, t.control = &addr, &dev, byUUID[bleSpeaker], byUUID[bleControl]
	h := t.h
	t.mu.Unlock()
	slog.Info("BLE connected", "to", addr.String())

	if err := byUUID[bleMic].EnableNotifications(func(b []byte) {
		if h.Audio != nil {
			h.Audio(append([]byte(nil), b...))
		}
	}); err != nil {
		return fmt.Errorf("ble: subscribe mic: %w", err)
	}
	return byUUID[bleControl].EnableNotifications(func(b []byte) {
		if e := Event(b[0]); len(b) > 0 && e.valid() && h.Event != nil {
			h.Event(e, append([]byte(nil), b[1:]...))
		}
	})
}

// scan finds the device by address, else by name among devices advertising
// the Ovi service, else the first such device.
func (t *BLE) scan() (bluetooth.Address, error) {
	slog.Info("Scanning for BLE device", "name", t.name, "address", t.address)
	var mu sync.Mutex
	var best, fallback *bluetooth.Address
	stop := time.AfterFunc(bleScanTimeout, func() { _ = t.adapter.StopScan() })
	defer stop.Stop()
	err := t.adapter.Scan(func(a *bluetooth.Adapter, r bluetooth.ScanResult) {
		mu.Lock()
		defer mu.Unlock()
		addr := r.Address
		switch {
		case t.address != "":
			if !strings.EqualFold(addr.String(), t.address) {
				return
			}
		case !r.HasServiceUUID(bleService):
			return
		case t.name != "" && !strings.Contains(strings.ToLower(r.LocalName()), strings.ToLower(t.name)):
			if fallback == nil {
				fallback = &addr
			}
			return
		}
		best = &addr
		_ = a.StopScan()
	})
	if err != nil {
		return bluetooth.Address{}, fmt.Errorf("ble: scan: %w", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if best == nil {
		best = fallback
	}
	if best == nil {
		return bluetooth.Address{}, fmt.Errorf("ble: no device found (name=%q address=%q)", t.name, t.address)
	}
	return *best, nil
}

// onLink is BlueZ's connection-state callback; a drop starts reconnecting.
func (t *BLE) onLink(dev bluetooth.Device, connected bool) {
	t.mu.Lock()
	lost := !connected && t.dev != nil && dev.Address.String() == t.dev.Address.String() && !t.closing
	if lost {
		t.dev, t.speaker, t.control = nil, nil, nil
	}
	h, ctx := t.h, t.ctx
	t.mu.Unlock()
	if !lost {
		return
	}
	slog.Warn("BLE link lost", "to", dev.Address.String())
	if h.Disconnect != nil {
		go h.Disconnect()
	}
	go reconnect(ctx, t.String(), bleRetry, t.dial, h)
}

func cmpOr(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
