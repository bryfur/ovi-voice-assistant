package device

import (
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"sync"
	"time"

	"tinygo.org/x/bluetooth"
)

// GATT UUIDs for the Ovi voice service.
const (
	serviceUUID = "BA5E0001-FADA-4C14-A34C-1AE0F0A0A0A0"
	audioTxUUID = "BA5E0002-FADA-4C14-A34C-1AE0F0A0A0A0" // mic → server (notify)
	audioRxUUID = "BA5E0003-FADA-4C14-A34C-1AE0F0A0A0A0" // server → speaker (write-no-resp)
	controlUUID = "BA5E0004-FADA-4C14-A34C-1AE0F0A0A0A0" // events (read/write/notify)
)

// BLE ATT MTU minus overhead — safe default for write-without-response.
const defaultMTUPayload = 512

// Reconnection parameters.
const (
	bleReconnectDelay = 2 * time.Second
	bleScanTimeout    = 10 * time.Second
)

// BLETransport is a transport over Bluetooth Low Energy.
type BLETransport struct {
	deviceName    string
	deviceAddress string
	mtuPayload    int
	autoReconnect bool

	adapter *bluetooth.Adapter

	mu        sync.Mutex
	address   *bluetooth.Address
	device    *bluetooth.Device
	audioRx   *bluetooth.DeviceCharacteristic
	control   *bluetooth.DeviceCharacteristic
	audioTx   *bluetooth.DeviceCharacteristic
	connected bool
	stopping  bool
	enabled   bool

	eventCB      EventCallback
	audioCB      AudioCallback
	disconnectCB DisconnectCallback
	connectCB    ConnectCallback
}

// NewBLETransport creates a BLE transport. Provide either deviceName (scan
// for it) or deviceAddress (connect directly).
func NewBLETransport(deviceName, deviceAddress string) (*BLETransport, error) {
	if deviceName == "" && deviceAddress == "" {
		return nil, errors.New("must provide device name or device address")
	}
	return &BLETransport{
		deviceName:    deviceName,
		deviceAddress: deviceAddress,
		mtuPayload:    defaultMTUPayload,
		autoReconnect: true,
		adapter:       bluetooth.DefaultAdapter,
	}, nil
}

// String implements Transport.
func (t *BLETransport) String() string {
	if t.deviceAddress != "" {
		return "ble:" + t.deviceAddress
	}
	return "ble:" + t.deviceName
}

// IsConnected implements Transport.
func (t *BLETransport) IsConnected() bool {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.connected && t.device != nil
}

// SetEventCallback implements Transport.
func (t *BLETransport) SetEventCallback(cb EventCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.eventCB = cb
}

// SetAudioCallback implements Transport.
func (t *BLETransport) SetAudioCallback(cb AudioCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.audioCB = cb
}

// SetDisconnectCallback implements Transport.
func (t *BLETransport) SetDisconnectCallback(cb DisconnectCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.disconnectCB = cb
}

// SetConnectCallback implements Transport.
func (t *BLETransport) SetConnectCallback(cb ConnectCallback) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.connectCB = cb
}

// Connect implements Transport.
func (t *BLETransport) Connect() error {
	t.mu.Lock()
	t.stopping = false
	t.mu.Unlock()
	if err := t.enable(); err != nil {
		return err
	}
	if t.address == nil {
		addr, err := t.scan()
		if err != nil {
			return err
		}
		t.address = &addr
	}
	return t.establish()
}

// Disconnect implements Transport.
func (t *BLETransport) Disconnect() error {
	t.mu.Lock()
	t.stopping = true
	t.connected = false
	dev := t.device
	t.device = nil
	t.mu.Unlock()
	if dev != nil {
		if err := dev.Disconnect(); err != nil {
			slog.Debug("BLE disconnect error", "err", err)
		}
		slog.Info("BLE transport disconnected")
	}
	return nil
}

// SendEvent implements Transport.
//
// BLE event protocol: [1 byte event_type][payload]
func (t *BLETransport) SendEvent(event EventType, payload []byte) error {
	t.mu.Lock()
	ch := t.control
	ok := t.connected && ch != nil
	t.mu.Unlock()
	if !ok {
		slog.Warn("Cannot send event — not connected", "event", event.String())
		return nil
	}
	data := append([]byte{byte(event)}, payload...)
	if _, err := ch.WriteWithoutResponse(data); err != nil {
		slog.Error("Failed to send event", "event", event.String(), "err", err)
		return err
	}
	return nil
}

// SendAudio implements Transport. Audio is chunked to fit within the
// BLE MTU.
func (t *BLETransport) SendAudio(data []byte) error {
	t.mu.Lock()
	ch := t.audioRx
	ok := t.connected && ch != nil
	t.mu.Unlock()
	if !ok {
		slog.Warn("Cannot send audio — not connected")
		return nil
	}
	for offset := 0; offset < len(data); offset += t.mtuPayload {
		end := offset + t.mtuPayload
		if end > len(data) {
			end = len(data)
		}
		if _, err := ch.WriteWithoutResponse(data[offset:end]); err != nil {
			slog.Error("Failed to send audio chunk", "err", err)
			return err
		}
	}
	return nil
}

// -- Internal --

func (t *BLETransport) enable() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.enabled {
		return nil
	}
	if err := t.adapter.Enable(); err != nil {
		return fmt.Errorf("enable BLE adapter: %w", err)
	}
	t.adapter.SetConnectHandler(t.onConnectionChange)
	t.enabled = true
	return nil
}

func mustUUID(s string) bluetooth.UUID {
	u, err := bluetooth.ParseUUID(strings.ToLower(s))
	if err != nil {
		panic(err)
	}
	return u
}

// scan finds the target BLE device by name or address.
func (t *BLETransport) scan() (bluetooth.Address, error) {
	slog.Info("Scanning for BLE device", "name", t.deviceName, "address", t.deviceAddress)
	svc := mustUUID(serviceUUID)

	type hit struct {
		addr bluetooth.Address
		name string
	}
	var (
		mu       sync.Mutex
		found    *hit
		fallback *hit
	)
	timer := time.AfterFunc(bleScanTimeout, func() { _ = t.adapter.StopScan() })
	defer timer.Stop()

	err := t.adapter.Scan(func(a *bluetooth.Adapter, r bluetooth.ScanResult) {
		mu.Lock()
		defer mu.Unlock()
		name := r.LocalName()
		if t.deviceAddress != "" {
			if strings.EqualFold(r.Address.String(), t.deviceAddress) {
				found = &hit{r.Address, name}
				_ = a.StopScan()
			}
			return
		}
		if !r.HasServiceUUID(svc) {
			return
		}
		if fallback == nil {
			fallback = &hit{r.Address, name}
		}
		if t.deviceName != "" && name != "" &&
			strings.Contains(strings.ToLower(name), strings.ToLower(t.deviceName)) {
			found = &hit{r.Address, name}
			_ = a.StopScan()
		}
	})
	if err != nil {
		return bluetooth.Address{}, fmt.Errorf("BLE scan: %w", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if found != nil {
		slog.Info("Found BLE device", "name", found.name, "address", found.addr.String())
		return found.addr, nil
	}
	if t.deviceAddress != "" {
		return bluetooth.Address{}, fmt.Errorf("BLE device with address %s not found", t.deviceAddress)
	}
	if fallback != nil {
		slog.Info("Found BLE device by service UUID", "name", fallback.name, "address", fallback.addr.String())
		return fallback.addr, nil
	}
	return bluetooth.Address{}, fmt.Errorf("no BLE device found (name=%q, service=%s)", t.deviceName, serviceUUID)
}

func (t *BLETransport) establish() error {
	if t.address == nil {
		return errors.New("no BLE device — call Connect() first")
	}
	slog.Info("Connecting to BLE device", "address", t.address.String())
	dev, err := t.adapter.Connect(*t.address, bluetooth.ConnectionParams{})
	if err != nil {
		return fmt.Errorf("BLE connect: %w", err)
	}

	services, err := dev.DiscoverServices([]bluetooth.UUID{mustUUID(serviceUUID)})
	if err != nil || len(services) == 0 {
		_ = dev.Disconnect()
		return fmt.Errorf("discover Ovi service: %w", err)
	}
	chars, err := services[0].DiscoverCharacteristics([]bluetooth.UUID{
		mustUUID(audioTxUUID), mustUUID(audioRxUUID), mustUUID(controlUUID),
	})
	if err != nil {
		_ = dev.Disconnect()
		return fmt.Errorf("discover characteristics: %w", err)
	}
	var audioTx, audioRx, control *bluetooth.DeviceCharacteristic
	for i := range chars {
		c := &chars[i]
		switch uuid := c.UUID().String(); {
		case strings.EqualFold(uuid, audioTxUUID):
			audioTx = c
		case strings.EqualFold(uuid, audioRxUUID):
			audioRx = c
		case strings.EqualFold(uuid, controlUUID):
			control = c
		}
	}
	if audioTx == nil || audioRx == nil || control == nil {
		_ = dev.Disconnect()
		return errors.New("ovi GATT characteristics not found")
	}

	t.mu.Lock()
	t.device = &dev
	t.audioTx = audioTx
	t.audioRx = audioRx
	t.control = control
	t.connected = true
	t.mu.Unlock()
	slog.Info("BLE connected", "address", t.address.String())

	if err := audioTx.EnableNotifications(t.onAudioNotify); err != nil {
		return fmt.Errorf("subscribe AUDIO_TX: %w", err)
	}
	slog.Debug("Subscribed to AUDIO_TX notifications")
	if err := control.EnableNotifications(t.onControlNotify); err != nil {
		return fmt.Errorf("subscribe CONTROL: %w", err)
	}
	slog.Debug("Subscribed to CONTROL notifications")
	return nil
}

func (t *BLETransport) onConnectionChange(dev bluetooth.Device, connected bool) {
	if connected {
		return
	}
	t.mu.Lock()
	if t.device == nil || dev.Address.String() != t.device.Address.String() {
		t.mu.Unlock()
		return
	}
	wasConnected := t.connected
	t.connected = false
	stopping := t.stopping
	disconnectCB := t.disconnectCB
	t.mu.Unlock()

	if stopping {
		return
	}
	if wasConnected {
		slog.Warn("BLE disconnected", "address", dev.Address.String())
	}
	if disconnectCB != nil {
		go disconnectCB()
	}
	if t.autoReconnect {
		go t.reconnectLoop()
	}
}

func (t *BLETransport) reconnectLoop() {
	for {
		time.Sleep(bleReconnectDelay)
		t.mu.Lock()
		stopping := t.stopping
		t.mu.Unlock()
		if stopping {
			return
		}
		slog.Info("BLE reconnecting", "address", t.address.String())
		// Re-scan in case the device address changed (e.g. random MAC).
		addr, err := t.scan()
		if err != nil {
			slog.Warn("BLE device not found during reconnect scan, retrying")
			continue
		}
		t.address = &addr
		if err := t.establish(); err != nil {
			slog.Warn("BLE reconnect failed, retrying", "delay", bleReconnectDelay, "err", err)
			continue
		}
		slog.Info("BLE reconnected successfully")
		t.mu.Lock()
		cb := t.connectCB
		t.mu.Unlock()
		if cb != nil {
			cb()
		}
		return
	}
}

// onAudioNotify handles incoming audio data from the device microphone.
func (t *BLETransport) onAudioNotify(buf []byte) {
	t.mu.Lock()
	cb := t.audioCB
	t.mu.Unlock()
	if cb != nil {
		data := make([]byte, len(buf))
		copy(data, buf)
		cb(data)
	}
}

// onControlNotify handles incoming control events from the device.
//
// BLE event protocol: [1 byte event_type][payload]
func (t *BLETransport) onControlNotify(buf []byte) {
	if len(buf) < 1 {
		slog.Warn("Received empty control notification")
		return
	}
	event := EventType(buf[0])
	if !event.Valid() {
		slog.Warn("Unknown BLE event type", "type", fmt.Sprintf("0x%02X", buf[0]))
		return
	}
	payload := make([]byte, len(buf)-1)
	copy(payload, buf[1:])
	t.mu.Lock()
	cb := t.eventCB
	t.mu.Unlock()
	if cb != nil {
		cb(event, payload)
	}
}
