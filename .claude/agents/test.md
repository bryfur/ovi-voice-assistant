# Test Agent

1. `go test ./...` — unit tests, no network and no models (fakes for the OpenAI API via `httptest`, a self-re-executing fake MCP server, fake VAD/STT/TTS, stub scripts for ffmpeg).
2. `go test -race ./internal/pipeline/ ./internal/device/ ./internal/music/` when touching concurrency.
3. `OVI_TEST_MODELS=1 go test ./internal/tts/ ./internal/stt/ -run Real` to exercise real Kokoro and Silero through sherpa-onnx (downloads once).

Tests are colocated (`foo_test.go`), use Arrange / Act / Assert with blank lines, and cover the success path, error path and cancellation where a context is involved. Fix the code when a test expresses intended behaviour; fix the test only when it encodes a wrong expectation.
