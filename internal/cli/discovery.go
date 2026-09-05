package cli

import (
	"cmp"
	"context"
	"net"
	"slices"
	"strings"
	"time"

	"github.com/grandcat/zeroconf"
)

// esphomeService is the mDNS service ESPHome devices advertise.
const esphomeService = "_esphomelib._tcp"

// Device is a discovered ESPHome device.
type Device struct {
	Name string
	Host string
	IP   string
	Port int
}

// mdnsBrowse browses for ESPHome devices; tests replace it.
var mdnsBrowse = func(ctx context.Context, entries chan<- *zeroconf.ServiceEntry) error {
	resolver, err := zeroconf.NewResolver(nil)
	if err != nil {
		return err
	}
	return resolver.Browse(ctx, esphomeService, "local.", entries)
}

// DiscoverDevices scans the local network for ESPHome devices.
func DiscoverDevices(timeout time.Duration) ([]Device, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	entries := make(chan *zeroconf.ServiceEntry, 32)
	if err := mdnsBrowse(ctx, entries); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	var devices []Device
scan:
	for {
		select {
		case <-ctx.Done():
			break scan
		case e, ok := <-entries:
			if !ok {
				break scan
			}
			ip := firstIP(e.AddrIPv4, e.AddrIPv6)
			if e == nil || ip == "" || seen[e.Instance] {
				continue
			}
			seen[e.Instance] = true
			name := strings.TrimSuffix(e.Instance, "."+esphomeService+".local.")
			devices = append(devices, Device{Name: name, Host: name + ".local", IP: ip, Port: cmp.Or(e.Port, 6055)})
		}
	}
	slices.SortFunc(devices, func(a, b Device) int { return strings.Compare(a.Name, b.Name) })
	return devices, nil
}

func firstIP(lists ...[]net.IP) string {
	for _, ips := range lists {
		if len(ips) > 0 {
			return ips[0].String()
		}
	}
	return ""
}
