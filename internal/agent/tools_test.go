package agent

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
	"github.com/bryfur/ovi-voice-assistant/internal/scheduler"
)

func findTool(t *testing.T, name string) Tool {
	t.Helper()
	for _, tool := range BuiltinTools() {
		if tool.Name == name {
			return tool
		}
	}
	t.Fatalf("tool %s not found", name)
	return Tool{}
}

func call(t *testing.T, name string, actx *Context, args Args) string {
	t.Helper()
	out, err := findTool(t, name).Handler(context.Background(), actx, args)
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	return out
}

func TestBuiltinToolsCountAndSayDisabled(t *testing.T) {
	tools := BuiltinTools()

	if len(tools) != 20 {
		t.Fatalf("expected 20 tools, got %d", len(tools))
	}
	if findTool(t, "say").Enabled {
		t.Fatal("say must be disabled")
	}
	for _, tool := range tools {
		if tool.Name != "say" && !tool.Enabled {
			t.Fatalf("%s should be enabled", tool.Name)
		}
	}
}

func TestSayTool(t *testing.T) {
	var spoken string
	actx := &Context{Say: func(_ context.Context, text string) error { spoken = text; return nil }}

	out := call(t, "say", actx, Args{"text": "hi"})
	none := call(t, "say", &Context{}, Args{"text": "hi"})

	if out != "Spoken." || spoken != "hi" || none != "Say not available." {
		t.Fatalf("got %q %q %q", out, spoken, none)
	}
}

func TestGetCurrentTime(t *testing.T) {
	out := call(t, "get_current_time", &Context{}, Args{})
	utc := call(t, "get_current_time", &Context{}, Args{"timezone": "UTC"})

	if !strings.Contains(out, " at ") || !strings.HasSuffix(utc, "UTC") {
		t.Fatalf("got %q / %q", out, utc)
	}
	if _, err := findTool(t, "get_current_time").Handler(context.Background(), &Context{}, Args{"timezone": "Nowhere/Land"}); err == nil {
		t.Fatal("expected error for bad timezone")
	}
}

func TestTimerTools(t *testing.T) {
	actx := &Context{}

	set := call(t, "set_timer", actx, Args{"minutes": 1.0, "seconds": 30.0, "label": "pasta"})
	check := call(t, "check_timer", actx, Args{})
	cancel := call(t, "cancel_timer", actx, Args{"label": "pasta"})
	missing := call(t, "cancel_timer", actx, Args{"label": "pasta"})
	zero := call(t, "set_timer", actx, Args{})
	none := call(t, "check_timer", actx, Args{})

	if set != "Timer 'pasta' set for 1 minute, 30 seconds." {
		t.Fatalf("set = %q", set)
	}
	if !strings.HasPrefix(check, "'pasta': 1m") || !strings.HasSuffix(check, "remaining") {
		t.Fatalf("check = %q", check)
	}
	if cancel != "Timer 'pasta' cancelled." || missing != "No active timer named 'pasta'." {
		t.Fatalf("cancel = %q / %q", cancel, missing)
	}
	if zero != "Timer duration must be greater than zero." || none != "No active timers." {
		t.Fatalf("zero = %q none = %q", zero, none)
	}
}

func TestSetTimerHours(t *testing.T) {
	actx := &Context{}

	out := call(t, "set_timer", actx, Args{"minutes": 125.0})

	if out != "Timer 'timer' set for 2 hours, 5 minutes." {
		t.Fatalf("got %q", out)
	}
	actx.CancelTimer("timer")
}

func TestCalculateTool(t *testing.T) {
	if out := call(t, "calculate", &Context{}, Args{"expression": "2 ** 10"}); out != "1024" {
		t.Fatalf("got %q", out)
	}
	if out := call(t, "calculate", &Context{}, Args{"expression": "x.y"}); !strings.HasPrefix(out, "Error:") {
		t.Fatalf("got %q", out)
	}
}

func TestRandomTools(t *testing.T) {
	one := call(t, "roll_dice", &Context{}, Args{})
	many := call(t, "roll_dice", &Context{}, Args{"sides": 6.0, "count": 3.0})
	num := call(t, "random_number", &Context{}, Args{"low": 5.0, "high": 5.0})
	coin := call(t, "flip_coin", &Context{}, Args{})

	if one < "1" || one > "6" || !strings.Contains(many, "(total: ") || num != "5" {
		t.Fatalf("got %q %q %q", one, many, num)
	}
	if coin != "Heads" && coin != "Tails" {
		t.Fatalf("coin = %q", coin)
	}
}

func TestUnitConvert(t *testing.T) {
	cases := []struct {
		value    float64
		from, to string
		want     string
	}{
		{10, "km", "miles", "6.2137 miles"},
		{100, "celsius", "fahrenheit", "212 fahrenheit"},
		{32, "F", "C", "0 C"},
		{1, "kg", "lbs", "2.2046 lbs"},
		{1, "parsecs", "km", "Unknown conversion: parsecs to km"},
	}

	for _, c := range cases {
		got := call(t, "unit_convert", &Context{}, Args{"value": c.value, "from_unit": c.from, "to_unit": c.to})

		if got != c.want {
			t.Errorf("unit_convert(%v %s→%s) = %q, want %q", c.value, c.from, c.to, got, c.want)
		}
	}
}

func stubSearch(t *testing.T, tracks []music.MusicTrack, err error) {
	t.Helper()
	old := music.SearchMusicFunc
	music.SearchMusicFunc = func(context.Context, string, string) ([]music.MusicTrack, error) { return tracks, err }
	t.Cleanup(func() { music.SearchMusicFunc = old })
}

func TestPlayMusicPerDevicePlayer(t *testing.T) {
	stubSearch(t, []music.MusicTrack{{Title: "Song", Artist: "Band"}, {Title: "B", Artist: "C"}}, nil)
	actx := &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}

	out := call(t, "play_music", actx, Args{"query": "song"})

	if out != "Playing 'Song' by Band and 1 more tracks." || !actx.MusicPlayer.IsActive() {
		t.Fatalf("got %q active=%v", out, actx.MusicPlayer.IsActive())
	}
}

func TestPlayMusicNoResultsAndNoPlayer(t *testing.T) {
	stubSearch(t, nil, nil)

	none := call(t, "play_music", &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}, Args{"query": "zzz"})
	stubSearch(t, []music.MusicTrack{{Title: "S", Artist: "A"}}, nil)
	unavailable := call(t, "play_music", &Context{}, Args{"query": "s"})

	if none != "No results found for 'zzz'." || unavailable != "Music playback is not available on this device." {
		t.Fatalf("got %q / %q", none, unavailable)
	}
}

func TestPlayMusicGroup(t *testing.T) {
	stubSearch(t, []music.MusicTrack{{Title: "Song", Artist: "Band"}}, nil)
	group := music.NewMusicGroup(48000, 2, nil)
	actx := &Context{MusicGroup: group}

	out := call(t, "play_music", actx, Args{"query": "song"})
	now := call(t, "now_playing", actx, Args{})
	pause := call(t, "pause_music", actx, Args{})
	resume := call(t, "resume_music", actx, Args{})
	skip := call(t, "skip_track", actx, Args{})
	stop := call(t, "stop_music", actx, Args{})

	if out != "Playing 'Song' by Band on all devices." {
		t.Fatalf("play = %q", out)
	}
	if now != "Currently playing: 'Song' by Band (track 1 of 1)." {
		t.Fatalf("now = %q", now)
	}
	if pause != "Paused 'Song' by Band." || resume != "Resuming 'Song' by Band." {
		t.Fatalf("pause/resume = %q / %q", pause, resume)
	}
	if skip != "No more tracks in the queue." || stop != "Music stopped." {
		t.Fatalf("skip/stop = %q / %q", skip, stop)
	}
	group.Close(context.Background())
}

func TestMusicControlsWithoutQueue(t *testing.T) {
	actx := &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}

	if call(t, "pause_music", actx, Args{}) != "No music is playing." ||
		call(t, "resume_music", actx, Args{}) != "No music to resume." ||
		call(t, "skip_track", actx, Args{}) != "No music is playing." ||
		call(t, "stop_music", actx, Args{}) != "No music is playing." ||
		call(t, "now_playing", actx, Args{}) != "No music is playing." {
		t.Fatal("empty-queue responses wrong")
	}
}

func TestPerDevicePlayerControls(t *testing.T) {
	actx := &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}
	actx.MusicPlayer.SetQueue([]music.MusicTrack{{Title: "A", Artist: "X"}, {Title: "B", Artist: "Y"}}, 0)

	pause := call(t, "pause_music", actx, Args{})
	now := call(t, "now_playing", actx, Args{})
	resume := call(t, "resume_music", actx, Args{})
	skip := call(t, "skip_track", actx, Args{})
	stop := call(t, "stop_music", actx, Args{})

	if pause != "Paused 'A' by X." || now != "Currently paused: 'A' by X (track 1 of 2)." {
		t.Fatalf("pause=%q now=%q", pause, now)
	}
	if resume != "Resuming 'A' by X." || skip != "Skipping to 'B' by Y." || stop != "Music stopped." {
		t.Fatalf("resume=%q skip=%q stop=%q", resume, skip, stop)
	}
}

func TestAutomationTools(t *testing.T) {
	sched := scheduler.New(filepath.Join(t.TempDir(), "a.json"), nil, nil)
	actx := &Context{Scheduler: sched}

	created := call(t, "create_automation", actx, Args{"name": "morning", "schedule": "0 7 * * *", "prompt": "weather?"})
	invalid := call(t, "create_automation", actx, Args{"name": "bad", "schedule": "nope", "prompt": "x"})
	list := call(t, "list_automations", actx, Args{})
	toggled := call(t, "toggle_automation", actx, Args{"name": "morning", "enabled": false})
	deleted := call(t, "delete_automation", actx, Args{"name": "morning"})
	missing := call(t, "delete_automation", actx, Args{"name": "morning"})
	empty := call(t, "list_automations", actx, Args{})

	if created != "Automation 'morning' created. Schedule: 0 7 * * *." || !strings.HasPrefix(invalid, "Invalid cron") {
		t.Fatalf("created=%q invalid=%q", created, invalid)
	}
	if !strings.Contains(list, "'morning' (enabled): schedule=0 7 * * *, prompt=\"weather?\"") {
		t.Fatalf("list = %q", list)
	}
	if toggled != "Automation 'morning' disabled." || deleted != "Automation 'morning' deleted." ||
		missing != "No automation named 'morning'." || empty != "No automations configured." {
		t.Fatalf("toggle=%q deleted=%q missing=%q empty=%q", toggled, deleted, missing, empty)
	}
}

func TestAutomationToolsUnavailable(t *testing.T) {
	for _, name := range []string{"create_automation", "list_automations", "delete_automation", "toggle_automation"} {
		if out := call(t, name, &Context{}, Args{"name": "x", "schedule": "* * * * *", "prompt": "p"}); out != "Automations are not available." {
			t.Fatalf("%s = %q", name, out)
		}
	}
}
