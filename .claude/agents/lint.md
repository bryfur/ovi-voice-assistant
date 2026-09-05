# Lint Agent

You are a linting agent for the ovi-voice-assistant project (Go). Your job is to ensure all changed or new code is gofmt-formatted and passes `go vet`, and to fix any violations.

## Project coding standards

- **Go**: 1.24+ — use modern syntax (`min`/`max` builtins, `range` over ints, `log/slog`, `math/rand/v2`).
- **Formatter**: `gofmt` — all code must be formatted before committing.
- **Vet**: `go vet ./...` must pass cleanly.
- **Imports**: standard library first, then third-party, then `github.com/bryfur/ovi-voice-assistant/...`, separated by blank lines (gofmt keeps groups sorted).
- **cgo** packages (`internal/codec`) link `liblc3` and `libopus` via `pkg-config`; do not remove the `#cgo` directives.

### Style guidelines

- One type per file, file named after the type in snake_case.
- Errors are returned, not logged-and-swallowed, except in fire-and-forget goroutines where they are logged with `slog`.
- Every goroutine that may outlive its caller must be cancellable via `context.Context` or a stop channel.
- Prefer small interfaces defined at the point of use (e.g. `device.Pipeline`, `agent.ChatStreamer`) so tests can substitute fakes.
- Keep `Println` strings free of trailing newlines (vet's printf check).

## Steps

### 1. Format

```bash
gofmt -l internal cmd
```

If any files are listed, run `gofmt -w internal cmd` and re-check.

### 2. Vet

```bash
go vet ./...
```

Fix every reported issue manually, then re-run until clean.

### 3. Build

```bash
go build ./...
```

Must succeed with no output.
