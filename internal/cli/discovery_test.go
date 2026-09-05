package cli

import (
	"context"
	"net"
	"testing"
	"time"

	"github.com/grandcat/zeroconf"
)

func TestDiscoverDevicesParsesEntries(t *testing.T) {
	old := mdnsBrowse
	defer func() { mdnsBrowse = old }()
	mdnsBrowse = func(ctx context.Context, entries chan<- *zeroconf.ServiceEntry) error {
		go func() {
			e := zeroconf.NewServiceEntry("voice-pe-1234", esphomeService, "local.")
			e.Port = 6055
			e.AddrIPv4 = []net.IP{net.ParseIP("192.168.1.10")}
			entries <- e
			dup := zeroconf.NewServiceEntry("voice-pe-1234", esphomeService, "local.")
			dup.AddrIPv4 = []net.IP{net.ParseIP("192.168.1.10")}
			entries <- dup
			noAddr := zeroconf.NewServiceEntry("ghost", esphomeService, "local.")
			entries <- noAddr
			close(entries)
		}()
		return nil
	}

	devices, err := DiscoverDevices(500 * time.Millisecond)

	if err != nil || len(devices) != 1 {
		t.Fatalf("got %+v, %v", devices, err)
	}
	d := devices[0]
	if d.Name != "voice-pe-1234" || d.Host != "voice-pe-1234.local" || d.IP != "192.168.1.10" || d.Port != 6055 {
		t.Fatalf("got %+v", d)
	}
}

func TestDiscoverDevicesTimesOut(t *testing.T) {
	old := mdnsBrowse
	defer func() { mdnsBrowse = old }()
	mdnsBrowse = func(ctx context.Context, entries chan<- *zeroconf.ServiceEntry) error { return nil }

	start := time.Now()
	devices, err := DiscoverDevices(100 * time.Millisecond)

	if err != nil || len(devices) != 0 || time.Since(start) > time.Second {
		t.Fatalf("got %v, %v after %v", devices, err, time.Since(start))
	}
}
