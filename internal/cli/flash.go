package cli

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"time"
)

// esphomeDir holds the device configs.
var esphomeDir = "esphome"

// preferredOrder sorts well-known device configs to the top.
var preferredOrder = []string{"voice-pe", "atom-echo", "s3-box-3"}

// deviceConfig is an ESPHome YAML available for flashing.
type deviceConfig struct {
	Path        string
	Name        string
	Description string
}

var (
	configForRe = regexp.MustCompile(`config for (.+)$`)
	bleForRe    = regexp.MustCompile(`— (.+) for (.+)$`)
	deviceName  = regexp.MustCompile(`(?m)^\s+name:\s+(\S+)`)
)

// findDeviceConfigs lists ESPHome device YAMLs with display names.
func findDeviceConfigs(dir string) []deviceConfig {
	paths, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
	sort.Strings(paths)
	var configs []deviceConfig
	for _, p := range paths {
		if filepath.Base(p) == "secrets.yaml" {
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		first := strings.SplitN(string(data), "\n", 2)[0]
		name := strings.TrimSuffix(filepath.Base(p), ".yaml")
		desc := name
		if m := configForRe.FindStringSubmatch(first); m != nil {
			desc = m[1]
		} else if m := bleForRe.FindStringSubmatch(first); m != nil {
			desc = fmt.Sprintf("%s (%s)", m[2], m[1])
		}
		configs = append(configs, deviceConfig{Path: p, Name: name, Description: desc})
	}
	rank := func(name string) int {
		for i, prefix := range preferredOrder {
			if strings.HasPrefix(name, prefix) {
				return i
			}
		}
		return len(preferredOrder)
	}
	sort.SliceStable(configs, func(i, j int) bool {
		ri, rj := rank(configs[i].Name), rank(configs[j].Name)
		if ri != rj {
			return ri < rj
		}
		return configs[i].Name < configs[j].Name
	})
	return configs
}

// checkSecrets reports whether secrets.yaml has real WiFi credentials and
// an encryption key.
func checkSecrets(path string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	s := string(data)
	hasWifi := strings.Contains(s, "wifi_ssid") && !strings.Contains(s, "my_wifi_ssid")
	return hasWifi && strings.Contains(s, "api_encryption_key")
}

// GenerateKey returns a base64-encoded 32-byte encryption key.
func GenerateKey() string {
	b := make([]byte, 32)
	_, _ = rand.Read(b)
	return base64.StdEncoding.EncodeToString(b)
}

// detectWifiSSID returns the currently connected WiFi SSID, if detectable.
func detectWifiSSID() string {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "linux":
		cmd = exec.Command("nmcli", "-t", "-f", "active,ssid", "dev", "wifi")
	case "darwin":
		cmd = exec.Command("/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport", "-I")
	default:
		return ""
	}
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	for _, line := range strings.Split(string(out), "\n") {
		if runtime.GOOS == "linux" && strings.HasPrefix(line, "yes:") {
			return strings.TrimPrefix(line, "yes:")
		}
		if runtime.GOOS == "darwin" && strings.Contains(line, " SSID:") {
			return strings.TrimSpace(strings.SplitN(line, ":", 2)[1])
		}
	}
	return ""
}

// writeSecrets writes secrets.yaml, preserving unrelated keys and an
// existing encryption key. It returns the key in use.
func writeSecrets(path, ssid, password string) (string, error) {
	existing := map[string]string{}
	var order []string
	if data, err := os.ReadFile(path); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if strings.Contains(line, ":") && !strings.HasPrefix(line, "#") {
				k := strings.TrimSpace(strings.SplitN(line, ":", 2)[0])
				existing[k] = line
				order = append(order, k)
			}
		}
	}
	lines := []string{
		fmt.Sprintf("wifi_ssid: %q", ssid),
		fmt.Sprintf("wifi_password: %q", password),
	}
	key := ""
	if line, ok := existing["api_encryption_key"]; ok {
		lines = append(lines, line)
		key = strings.Trim(strings.TrimSpace(strings.SplitN(line, ":", 2)[1]), `"'`)
	} else {
		key = GenerateKey()
		lines = append(lines, fmt.Sprintf("api_encryption_key: %q", key))
	}
	for _, k := range order {
		if k != "wifi_ssid" && k != "wifi_password" && k != "api_encryption_key" {
			lines = append(lines, existing[k])
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return key, os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600)
}

// serialPort is a candidate serial device.
type serialPort struct {
	Device      string
	Description string
}

// detectSerialPorts lists USB serial devices (generic ttyS* ports excluded).
func detectSerialPorts() []serialPort {
	var ports []serialPort
	var globs []string
	switch runtime.GOOS {
	case "linux":
		globs = []string{"/dev/ttyUSB*", "/dev/ttyACM*"}
	case "darwin":
		globs = []string{"/dev/cu.usbserial*", "/dev/cu.usbmodem*", "/dev/cu.SLAB*", "/dev/cu.wchusbserial*"}
	}
	for _, g := range globs {
		matches, _ := filepath.Glob(g)
		for _, m := range matches {
			ports = append(ports, serialPort{Device: m, Description: m})
		}
	}
	sort.Slice(ports, func(i, j int) bool { return ports[i].Device < ports[j].Device })
	return ports
}

// esphomeDeviceName extracts the ESPHome device name from a YAML config.
func esphomeDeviceName(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	if m := deviceName.FindStringSubmatch(string(data)); m != nil {
		return m[1]
	}
	return ""
}

// esphomeCommand finds an esphome launcher: `esphome` on PATH, else `uv run esphome`.
func esphomeCommand(args ...string) *exec.Cmd {
	if _, err := exec.LookPath("esphome"); err == nil {
		return exec.Command("esphome", args...)
	}
	return exec.Command("uv", append([]string{"run", "esphome"}, args...)...)
}

// Flash drives the interactive device flashing flow.
func Flash(c *IO) {
	c.Println()
	c.Panel("Ovi — Device Flashing", "Compile and flash ESPHome firmware to a device")

	if st, err := os.Stat(esphomeDir); err != nil || !st.IsDir() {
		c.Print("  ESPHome directory not found: %s\n  Run this command from the Ovi project root.\n", esphomeDir)
		return
	}
	configs := findDeviceConfigs(esphomeDir)
	if len(configs) == 0 {
		c.Print("  No device configs found in %s/\n", esphomeDir)
		return
	}

	secretsPath := filepath.Join(esphomeDir, "secrets.yaml")
	if !checkSecrets(secretsPath) {
		c.Print("  WiFi credentials not configured.\n\n")
		if !promptSecrets(c, secretsPath) {
			c.Println("  WiFi credentials required for flashing.")
			return
		}
	}

	c.Print("\n  Select a device to flash:\n\n")
	for i, cfg := range configs {
		c.Print("    %d. %s (%s.yaml)\n", i+1, cfg.Description, cfg.Name)
	}
	var selected deviceConfig
	for {
		raw := c.Prompt("\n  Device number", "", false)
		var idx int
		if _, err := fmt.Sscanf(raw, "%d", &idx); err == nil && idx >= 1 && idx <= len(configs) {
			selected = configs[idx-1]
			break
		}
		c.Print("  Enter a number 1-%d\n", len(configs))
	}
	c.Print("\n  Selected: %s\n", selected.Description)

	c.Print("\n  Flash method:\n\n")
	c.Println("    1. USB — flash over serial (first time or recovery)")
	c.Println("    2. OTA — flash over WiFi (device already running)")
	method := c.Choice("\n  Method", []string{"1", "2"}, "1")

	args := []string{"run", selected.Path, "--no-logs"}
	if method == "2" {
		args = append(args, "--device", "OTA")
		c.Println("\n  Using OTA — device must be on the network.")
	} else {
		ports := detectSerialPorts()
		if len(ports) > 0 {
			c.Print("\n  Serial ports detected:\n\n")
			for i, p := range ports {
				c.Print("    %d. %s — %s\n", i+1, p.Device, p.Description)
			}
			for {
				raw := c.Prompt("\n  Port number (or Enter for auto-detect)", "", false)
				if raw == "" {
					break
				}
				var idx int
				if _, err := fmt.Sscanf(raw, "%d", &idx); err == nil && idx >= 1 && idx <= len(ports) {
					args = append(args, "--device", ports[idx-1].Device)
					break
				}
				c.Print("  Enter a number 1-%d\n", len(ports))
			}
		} else {
			c.Println("\n  No serial ports detected.")
			c.Println("  Make sure the device is plugged in via USB.")
			if !c.Confirm("  Continue anyway?", true) {
				return
			}
		}
	}

	c.Println()
	c.Rule("Compiling and flashing")
	cmd := esphomeCommand(args...)
	c.Print("  Running: %s\n\n", strings.Join(cmd.Args, " "))
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	err := cmd.Run()

	c.Println()
	if err == nil {
		c.Println("  Flash complete!")
		c.Print("  The device will reboot and connect to WiFi.\n\n")
		if name := esphomeDeviceName(selected.Path); name != "" {
			scanAndAddDevice(c, name)
		}
		return
	}
	c.Print("  Flash failed (%v)\n", err)
	if method == "1" {
		c.Println("  Tips:\n  - Hold the BOOT button while plugging in USB\n" +
			"  - Check that the serial port is not in use\n" +
			"  - Try a different USB cable (data, not charge-only)")
	}
}

func promptSecrets(c *IO, path string) bool {
	c.Print("  WiFi credentials are needed for device firmware.\n  They will be saved to %s\n\n", path)
	ssid := c.Prompt("  WiFi SSID", detectWifiSSID(), true)
	password := c.PromptHidden("  WiFi password", "")
	if ssid == "" {
		return false
	}
	key, err := writeSecrets(path, ssid, password)
	if err != nil {
		c.Print("  Failed to write secrets: %v\n", err)
		return false
	}
	c.Print("  API encryption key: %s\n", key)
	c.Print("  Saved to %s\n\n", path)
	return true
}

func scanAndAddDevice(c *IO, name string) {
	c.Println("  Scanning for the device on the network...")
	devices, err := DiscoverDevices(10 * time.Second)
	if err != nil {
		c.Print("  Scan failed: %v\n", err)
		return
	}
	for _, d := range devices {
		if strings.HasPrefix(d.Name, name) {
			c.Print("  Found: %s (%s)\n", d.Name, d.IP)
			if err := AddDevicesToConfig([]string{d.Host}, ""); err != nil {
				c.Print("  Failed to update config: %v\n", err)
				return
			}
			c.Print("  Added %s to config\n", d.Host)
			return
		}
	}
	c.Println("  Device not found on the network yet.\n  It may still be booting. Run ovi --scan in a moment.")
}
