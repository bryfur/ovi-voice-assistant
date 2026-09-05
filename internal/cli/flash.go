package cli

import (
	"cmp"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"time"
)

// esphomeDir holds the device configs.
var esphomeDir = "esphome"

// preferredOrder lists the device configs shown first.
var preferredOrder = []string{"voice-pe", "atom-echo", "s3-box-3"}

// deviceConfig is an ESPHome YAML that can be flashed.
type deviceConfig struct {
	Path, Name, Description string
}

var (
	configForRe = regexp.MustCompile(`config for (.+)$`)
	bleForRe    = regexp.MustCompile(`— (.+) for (.+)$`)
	deviceName  = regexp.MustCompile(`(?m)^\s+name:\s+(\S+)`)
)

// findDeviceConfigs lists the device YAMLs, well-known boards first.
func findDeviceConfigs(dir string) []deviceConfig {
	paths, _ := filepath.Glob(filepath.Join(dir, "*.yaml"))
	var configs []deviceConfig
	for _, p := range paths {
		data, err := os.ReadFile(p)
		if filepath.Base(p) == "secrets.yaml" || err != nil {
			continue
		}
		first, _, _ := strings.Cut(string(data), "\n")
		name := strings.TrimSuffix(filepath.Base(p), ".yaml")
		desc := name
		if m := configForRe.FindStringSubmatch(first); m != nil {
			desc = m[1]
		} else if m := bleForRe.FindStringSubmatch(first); m != nil {
			desc = fmt.Sprintf("%s (%s)", m[2], m[1])
		}
		configs = append(configs, deviceConfig{Path: p, Name: name, Description: desc})
	}
	rank := func(c deviceConfig) int {
		i := slices.IndexFunc(preferredOrder, func(prefix string) bool { return strings.HasPrefix(c.Name, prefix) })
		if i < 0 {
			return len(preferredOrder)
		}
		return i
	}
	slices.SortStableFunc(configs, func(a, b deviceConfig) int {
		return cmp.Or(cmp.Compare(rank(a), rank(b)), strings.Compare(a.Name, b.Name))
	})
	return configs
}

// checkSecrets reports whether the secrets file has real WiFi credentials
// and an encryption key.
func checkSecrets(path string) bool {
	data, err := os.ReadFile(path)
	s := string(data)
	return err == nil && strings.Contains(s, "wifi_ssid") && !strings.Contains(s, "my_wifi_ssid") && strings.Contains(s, "api_encryption_key")
}

// GenerateKey returns a base64-encoded 32-byte encryption key.
func GenerateKey() string {
	b := make([]byte, 32)
	_, _ = rand.Read(b)
	return base64.StdEncoding.EncodeToString(b)
}

// detectWifiSSID returns the WiFi network this machine is on, if known.
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
	for line := range strings.SplitSeq(string(out), "\n") {
		if ssid, ok := strings.CutPrefix(line, "yes:"); ok && runtime.GOOS == "linux" {
			return ssid
		}
		if _, ssid, ok := strings.Cut(line, " SSID:"); ok && runtime.GOOS == "darwin" {
			return strings.TrimSpace(ssid)
		}
	}
	return ""
}

// writeSecrets writes the secrets file with the WiFi credentials, keeping
// an existing encryption key and any other entries. It returns the key.
func writeSecrets(path, ssid, password string) (string, error) {
	kept := map[string]string{}
	var order []string
	if data, err := os.ReadFile(path); err == nil {
		for line := range strings.SplitSeq(string(data), "\n") {
			if k, _, ok := strings.Cut(line, ":"); ok && !strings.HasPrefix(line, "#") {
				kept[strings.TrimSpace(k)] = line
				order = append(order, strings.TrimSpace(k))
			}
		}
	}
	key := GenerateKey()
	if line, ok := kept["api_encryption_key"]; ok {
		_, value, _ := strings.Cut(line, ":")
		key = strings.Trim(strings.TrimSpace(value), `"'`)
	}
	lines := []string{fmt.Sprintf("wifi_ssid: %q", ssid), fmt.Sprintf("wifi_password: %q", password), fmt.Sprintf("api_encryption_key: %q", key)}
	for _, k := range order {
		if k != "wifi_ssid" && k != "wifi_password" && k != "api_encryption_key" {
			lines = append(lines, kept[k])
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return "", err
	}
	return key, os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600)
}

// serialPorts lists USB serial devices, sorted.
func serialPorts() []string {
	var globs []string
	switch runtime.GOOS {
	case "linux":
		globs = []string{"/dev/ttyUSB*", "/dev/ttyACM*"}
	case "darwin":
		globs = []string{"/dev/cu.usbserial*", "/dev/cu.usbmodem*", "/dev/cu.SLAB*", "/dev/cu.wchusbserial*"}
	}
	var ports []string
	for _, g := range globs {
		matches, _ := filepath.Glob(g)
		ports = append(ports, matches...)
	}
	slices.Sort(ports)
	return ports
}

// esphomeDeviceName reads the device name from an ESPHome YAML.
func esphomeDeviceName(path string) string {
	data, err := os.ReadFile(path)
	if m := deviceName.FindSubmatch(data); err == nil && m != nil {
		return string(m[1])
	}
	return ""
}

// esphomeCommand runs esphome from PATH, else through uv.
func esphomeCommand(args ...string) *exec.Cmd {
	if _, err := exec.LookPath("esphome"); err == nil {
		return exec.Command("esphome", args...)
	}
	return exec.Command("uv", append([]string{"run", "esphome"}, args...)...)
}

// pickNumber asks for a 1-based index into n items; 0 means none chosen.
func pickNumber(c *Console, label string, n int, optional bool) int {
	for {
		raw := c.Prompt(label, "", false)
		if raw == "" && optional {
			return 0
		}
		if i, err := strconv.Atoi(raw); err == nil && i >= 1 && i <= n {
			return i
		}
		c.Print("  Enter a number 1-%d\n", n)
	}
}

// Flash drives the interactive device flashing flow.
func Flash(c *Console) {
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
	secrets := filepath.Join(esphomeDir, "secrets.yaml")
	if !checkSecrets(secrets) {
		c.Print("  WiFi credentials not configured.\n\n")
		if !promptSecrets(c, secrets) {
			c.Println("  WiFi credentials required for flashing.")
			return
		}
	}

	c.Print("\n  Select a device to flash:\n\n")
	for i, cfg := range configs {
		c.Print("    %d. %s (%s.yaml)\n", i+1, cfg.Description, cfg.Name)
	}
	selected := configs[pickNumber(c, "\n  Device number", len(configs), false)-1]
	c.Print("\n  Selected: %s\n", selected.Description)

	c.Print("\n  Flash method:\n\n")
	c.Println("    1. USB — flash over serial (first time or recovery)")
	c.Println("    2. OTA — flash over WiFi (device already running)")
	usb := c.Choice("\n  Method", []string{"1", "2"}, "1") == "1"
	args := []string{"run", selected.Path, "--no-logs"}
	switch ports := serialPorts(); {
	case !usb:
		args = append(args, "--device", "OTA")
		c.Println("\n  Using OTA — device must be on the network.")
	case len(ports) > 0:
		c.Print("\n  Serial ports detected:\n\n")
		for i, p := range ports {
			c.Print("    %d. %s\n", i+1, p)
		}
		if i := pickNumber(c, "\n  Port number (or Enter for auto-detect)", len(ports), true); i > 0 {
			args = append(args, "--device", ports[i-1])
		}
	default:
		c.Println("\n  No serial ports detected.\n  Make sure the device is plugged in via USB.")
		if !c.Confirm("  Continue anyway?", true) {
			return
		}
	}

	c.Println()
	c.Rule("Compiling and flashing")
	cmd := esphomeCommand(args...)
	c.Print("  Running: %s\n\n", strings.Join(cmd.Args, " "))
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	err := cmd.Run()
	c.Println()
	if err != nil {
		c.Print("  Flash failed (%v)\n", err)
		if usb {
			c.Println("  Tips:\n  - Hold the BOOT button while plugging in USB\n" +
				"  - Check that the serial port is not in use\n" +
				"  - Try a different USB cable (data, not charge-only)")
		}
		return
	}
	c.Println("  Flash complete!")
	c.Print("  The device will reboot and connect to WiFi.\n\n")
	if name := esphomeDeviceName(selected.Path); name != "" {
		scanAndAddDevice(c, name)
	}
}

func promptSecrets(c *Console, path string) bool {
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
	c.Print("  API encryption key: %s\n  Saved to %s\n\n", key, path)
	return true
}

func scanAndAddDevice(c *Console, name string) {
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
