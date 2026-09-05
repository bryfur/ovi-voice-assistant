package cli

import (
	"context"
	"sort"
	"strings"
	"time"

	"github.com/grandcat/zeroconf"
)

// esphomeService is the mDNS service advertised by ESPHome devices.
const esphomeService = "_esphomelib._tcp"

// Device is a discovered ESPHome device.
type Device struct {
	Name string
	Host string
	IP   string
	Port int
}

// mdnsBrowse is the mDNS browse implementation; tests may replace it.
var mdnsBrowse = func(ctx context.Context, entries chan<- *zeroconf.ServiceEntry) error {
	resolver, err := zeroconf.NewResolver(nil)
	if err != nil {
		return err
	}
	return resolver.Browse(ctx, esphomeService, "local.", entries)
}

// DiscoverDevices scans the local network for ESPHome devices for timeout.
func DiscoverDevices(timeout time.Duration) ([]Device, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	entries := make(chan *zeroconf.ServiceEntry, 32)
	if err := mdnsBrowse(ctx, entries); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var devices []Device
	for {
		select {
		case <-ctx.Done():
			sort.Slice(devices, func(i, j int) bool { return devices[i].Name < devices[j].Name })
			return devices, nil
		case e, ok := <-entries:
			if !ok {
				sort.Slice(devices, func(i, j int) bool { return devices[i].Name < devices[j].Name })
				return devices, nil
			}
			if e == nil || seen[e.Instance] {
				continue
			}
			ip := ""
			if len(e.AddrIPv4) > 0 {
				ip = e.AddrIPv4[0].String()
			} else if len(e.AddrIPv6) > 0 {
				ip = e.AddrIPv6[0].String()
			}
			if ip == "" {
				continue
			}
			seen[e.Instance] = true
			port := e.Port
			if port == 0 {
				port = 6055
			}
			name := strings.TrimSuffix(e.Instance, "."+esphomeService+".local.")
			devices = append(devices, Device{Name: name, Host: name + ".local", IP: ip, Port: port})
		}
	}
}
