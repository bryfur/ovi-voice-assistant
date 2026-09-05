package ort

import (
	"runtime"
	"strings"
	"testing"
)

func TestReleaseAssetForCurrentPlatform(t *testing.T) {
	asset, err := releaseAsset()

	if runtime.GOOS == "linux" || runtime.GOOS == "darwin" {
		if err != nil || !strings.Contains(asset, Version) || !strings.HasSuffix(asset, ".tgz") {
			t.Fatalf("asset=%q err=%v", asset, err)
		}
	}
}

func TestSelectProviders(t *testing.T) {
	if p := SelectProviders("cuda", true); len(p) != 1 || p[0] != ProviderCUDA {
		t.Fatalf("cuda = %v", p)
	}
	if p := SelectProviders("auto", false); len(p) == 0 || p[0] != ProviderCUDA {
		t.Fatalf("auto = %v", p)
	}
	if runtime.GOOS == "linux" {
		if p := SelectProviders("cpu", true); len(p) != 0 {
			t.Fatalf("cpu on linux = %v", p)
		}
	}
}

func TestVersionMatchesBindingAPI(t *testing.T) {
	if !strings.HasPrefix(Version, "1.29.") {
		t.Fatalf("Version %s must match the ORT_API_VERSION 29 headers in onnxruntime_go", Version)
	}
}
