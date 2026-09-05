# Lint Agent

Keep the Go tree formatted, vetted and simple.

1. `gofmt -l internal cmd` must print nothing (`gofmt -w` to fix).
2. `go vet ./...` must pass.
3. `go build ./...` must succeed.
4. Optional: `go run honnef.co/go/tools/cmd/staticcheck@latest ./...` should be clean.

Style: one type per file named after the type; lowercase error strings; `slog` logging; no trailing newlines in `Println` strings; delete unused code rather than keeping it "just in case"; keep the seven packages and unexport anything that does not cross a package line. The cgo codecs (`internal/device`) need `liblc3`/`libopus` via pkg-config; sherpa-onnx is prebuilt in the module cache.
