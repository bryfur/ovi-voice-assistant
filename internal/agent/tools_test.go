package agent

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/agent/scheduler"
	"github.com/bryfur/ovi-voice-assistant/internal/device"
	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

func call(t *testing.T, name string, env *Env, args Args) string {
	t.Helper()
	for _, tool := range builtinTools() {
		if tool.Name == name {
			out, err := tool.Run(ctx, env, args)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			return out
		}
	}
	t.Fatalf("tool %s not found", name)
	return ""
}

func TestBuiltinToolCount(t *testing.T) {
	if n := len(builtinTools()); n != 19 {
		t.Fatalf("expected 19 tools, got %d", n)
	}
}

func TestGetCurrentTime(t *testing.T) {
	local := call(t, "get_current_time", &Env{}, Args{})
	utc := call(t, "get_current_time", &Env{}, Args{"timezone": "UTC"})

	if !strings.Contains(local, " at ") || !strings.HasSuffix(utc, "UTC") {
		t.Fatalf("got %q / %q", local, utc)
	}
}

func TestTimerTools(t *testing.T) {
	env := &Env{}

	set := call(t, "set_timer", env, Args{"minutes": 1.0, "seconds": 30.0, "label": "pasta"})
	check := call(t, "check_timer", env, Args{})
	cancel := call(t, "cancel_timer", env, Args{"label": "pasta"})
	missing := call(t, "cancel_timer", env, Args{"label": "pasta"})
	zero := call(t, "set_timer", env, Args{})
	hours := call(t, "set_timer", env, Args{"minutes": 125.0})
	env.CancelTimer("timer")

	if set != "Timer 'pasta' set for 1 minute, 30 seconds." || !strings.HasPrefix(check, "'pasta': 1 minute, 2") {
		t.Fatalf("set=%q check=%q", set, check)
	}
	if cancel != "Timer 'pasta' cancelled." || missing != "No active timer named 'pasta'." {
		t.Fatalf("cancel=%q missing=%q", cancel, missing)
	}
	if zero != "Timer duration must be greater than zero." || hours != "Timer 'timer' set for 2 hours, 5 minutes." {
		t.Fatalf("zero=%q hours=%q", zero, hours)
	}
	if call(t, "check_timer", env, Args{}) != "No active timers." || spoken(0) != "0 seconds" {
		t.Fatal("expected no timers")
	}
}

func TestCalculateAndRandomTools(t *testing.T) {
	if out := call(t, "calculate", &Env{}, Args{"expression": "2 ** 10"}); out != "1024" {
		t.Fatalf("got %q", out)
	}
	if out := call(t, "calculate", &Env{}, Args{"expression": "x.y"}); !strings.HasPrefix(out, "Error:") {
		t.Fatalf("got %q", out)
	}
	one := call(t, "roll_dice", &Env{}, Args{})
	many := call(t, "roll_dice", &Env{}, Args{"sides": 6.0, "count": 3.0})
	num := call(t, "random_number", &Env{}, Args{"low": 5.0, "high": 5.0})
	coin := call(t, "flip_coin", &Env{}, Args{})
	if one < "1" || one > "6" || !strings.Contains(many, "(total: ") || num != "5" || (coin != "Heads" && coin != "Tails") {
		t.Fatalf("got %q %q %q %q", one, many, num, coin)
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
		{1, "parsecs", "km", "Unknown conversion: parsecs to km"},
	}
	for _, c := range cases {
		got := call(t, "unit_convert", &Env{}, Args{"value": c.value, "from_unit": c.from, "to_unit": c.to})

		if got != c.want {
			t.Errorf("%v %s to %s = %q, want %q", c.value, c.from, c.to, got, c.want)
		}
	}
}

// fakeMusic records searches and returns scripted tracks.
type fakeMusic struct {
	name   string
	tracks []music.Track
	seen   *[]string
}

func (f fakeMusic) Search(context.Context, string, int) ([]music.Track, error) {
	*f.seen = append(*f.seen, f.name)
	return f.tracks, nil
}

func (f fakeMusic) Play(context.Context, music.Track, device.Output) error { return nil }

func musicEnv(tracks ...music.Track) (*Env, *[]string) {
	seen := &[]string{}
	p := music.NewPlayer(map[string]music.Service{
		"youtube": fakeMusic{"youtube", tracks, seen},
		"spotify": fakeMusic{"spotify", tracks, seen},
	})
	return &Env{Music: p}, seen
}

func TestPlayMusicRoutesToTheService(t *testing.T) {
	env, seen := musicEnv(music.Track{Title: "Song", Artist: "Band"}, music.Track{Title: "B", Artist: "C"})

	out := call(t, "play_music", env, Args{"query": "song", "service": "spotify"})
	none := call(t, "play_music", env, Args{"query": "zzz", "service": "tidal"})

	if out != "Playing 'Song' by Band and 1 more tracks." || env.Music.Current().Title != "Song" {
		t.Fatalf("out=%q", out)
	}
	if !strings.HasPrefix(none, "Music search failed") || (*seen)[0] != "spotify" {
		t.Fatalf("none=%q seen=%v", none, *seen)
	}
}

func TestPlayMusicDefaultsToYouTube(t *testing.T) {
	env, seen := musicEnv()

	out := call(t, "play_music", env, Args{"query": "zzz"})

	if out != "No results found for 'zzz'." || (*seen)[0] != "youtube" {
		t.Fatalf("out=%q seen=%v", out, *seen)
	}
}

func TestMusicControls(t *testing.T) {
	env, _ := musicEnv(music.Track{Title: "Song", Artist: "Band"})

	play := call(t, "play_music", env, Args{"query": "song"})
	now := call(t, "now_playing", env, Args{})
	pause := call(t, "pause_music", env, Args{})
	paused := call(t, "now_playing", env, Args{})
	resume := call(t, "resume_music", env, Args{})
	skip := call(t, "skip_track", env, Args{})
	stop := call(t, "stop_music", env, Args{})

	if play != "Playing 'Song' by Band." || now != "Currently playing: 'Song' by Band (0 more queued)." || paused != "Currently paused: 'Song' by Band (0 more queued)." {
		t.Fatalf("play=%q now=%q paused=%q", play, now, paused)
	}
	if pause != "Paused 'Song' by Band." || resume != "Resuming 'Song' by Band." || skip != "No more tracks in the queue." || stop != "No music is playing." {
		t.Fatalf("pause=%q resume=%q skip=%q stop=%q", pause, resume, skip, stop)
	}
}

func TestMusicControlsWithoutMusic(t *testing.T) {
	env, _ := musicEnv()
	for name, want := range map[string]string{
		"pause_music": "No music is playing.", "resume_music": "No music to resume.", "skip_track": "No music is playing.",
		"stop_music": "No music is playing.", "now_playing": "No music is playing.",
	} {
		if got := call(t, name, env, Args{}); got != want {
			t.Errorf("%s = %q", name, got)
		}
	}
	if got := call(t, "play_music", &Env{}, Args{"query": "s"}); got != "Music playback is not available." {
		t.Fatalf("got %q", got)
	}
}

func TestAutomationTools(t *testing.T) {
	env := &Env{Scheduler: scheduler.New(filepath.Join(t.TempDir(), "a.json"), nil, nil)}

	created := call(t, "create_automation", env, Args{"name": "morning", "schedule": "0 7 * * *", "prompt": "weather?"})
	invalid := call(t, "create_automation", env, Args{"name": "bad", "schedule": "nope", "prompt": "x"})
	list := call(t, "list_automations", env, Args{})
	toggled := call(t, "toggle_automation", env, Args{"name": "morning", "enabled": false})
	deleted := call(t, "delete_automation", env, Args{"name": "morning"})
	missing := call(t, "delete_automation", env, Args{"name": "morning"})

	if created != "Automation 'morning' created. Schedule: 0 7 * * *." || !strings.HasPrefix(invalid, "invalid cron") {
		t.Fatalf("created=%q invalid=%q", created, invalid)
	}
	if !strings.Contains(list, "'morning' (enabled)") || toggled != "Automation 'morning' disabled." ||
		deleted != "Automation 'morning' deleted." || missing != "No automation named 'morning'." {
		t.Fatalf("list=%q toggled=%q deleted=%q missing=%q", list, toggled, deleted, missing)
	}
	if call(t, "list_automations", &Env{}, Args{}) != noAutomations || call(t, "list_automations", env, Args{}) != "No automations configured." {
		t.Fatal("expected unavailable / empty")
	}
}
