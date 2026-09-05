package agent

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

func call(t *testing.T, name string, actx *Context, args Args) string {
	t.Helper()
	for _, tool := range builtinTools() {
		if tool.Name == name {
			out, err := tool.Handler(context.Background(), actx, args)
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
	out := call(t, "get_current_time", &Context{}, Args{})
	utc := call(t, "get_current_time", &Context{}, Args{"timezone": "UTC"})

	if !strings.Contains(out, " at ") || !strings.HasSuffix(utc, "UTC") {
		t.Fatalf("got %q / %q", out, utc)
	}
}

func TestTimerTools(t *testing.T) {
	actx := &Context{}

	set := call(t, "set_timer", actx, Args{"minutes": 1.0, "seconds": 30.0, "label": "pasta"})
	check := call(t, "check_timer", actx, Args{})
	cancel := call(t, "cancel_timer", actx, Args{"label": "pasta"})
	missing := call(t, "cancel_timer", actx, Args{"label": "pasta"})
	zero := call(t, "set_timer", actx, Args{})
	hours := call(t, "set_timer", actx, Args{"minutes": 125.0})
	actx.CancelTimer("timer")

	if set != "Timer 'pasta' set for 1 minute, 30 seconds." || !strings.HasPrefix(check, "'pasta': 1m") {
		t.Fatalf("set=%q check=%q", set, check)
	}
	if cancel != "Timer 'pasta' cancelled." || missing != "No active timer named 'pasta'." {
		t.Fatalf("cancel=%q missing=%q", cancel, missing)
	}
	if zero != "Timer duration must be greater than zero." || hours != "Timer 'timer' set for 2 hours, 5 minutes." {
		t.Fatalf("zero=%q hours=%q", zero, hours)
	}
	if call(t, "check_timer", actx, Args{}) != "No active timers." {
		t.Fatal("expected no timers")
	}
}

func TestCalculateAndRandomTools(t *testing.T) {
	if out := call(t, "calculate", &Context{}, Args{"expression": "2 ** 10"}); out != "1024" {
		t.Fatalf("got %q", out)
	}
	if out := call(t, "calculate", &Context{}, Args{"expression": "x.y"}); !strings.HasPrefix(out, "Error:") {
		t.Fatalf("got %q", out)
	}
	one := call(t, "roll_dice", &Context{}, Args{})
	many := call(t, "roll_dice", &Context{}, Args{"sides": 6.0, "count": 3.0})
	num := call(t, "random_number", &Context{}, Args{"low": 5.0, "high": 5.0})
	coin := call(t, "flip_coin", &Context{}, Args{})
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
		got := call(t, "unit_convert", &Context{}, Args{"value": c.value, "from_unit": c.from, "to_unit": c.to})

		if got != c.want {
			t.Errorf("%v %s→%s = %q, want %q", c.value, c.from, c.to, got, c.want)
		}
	}
}

func stubSearch(t *testing.T, tracks []music.MusicTrack) *string {
	t.Helper()
	var service string
	old := music.SearchMusicFunc
	music.SearchMusicFunc = func(_ context.Context, _, svc string) ([]music.MusicTrack, error) {
		service = svc
		return tracks, nil
	}
	t.Cleanup(func() { music.SearchMusicFunc = old })
	return &service
}

func TestPlayMusicRoutesServiceAndUsesPlayer(t *testing.T) {
	service := stubSearch(t, []music.MusicTrack{{Title: "Song", Artist: "Band"}, {Title: "B", Artist: "C"}})
	actx := &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}

	out := call(t, "play_music", actx, Args{"query": "song", "service": "spotify"})

	if out != "Playing 'Song' by Band and 1 more tracks." || *service != "spotify" || !actx.MusicPlayer.IsActive() {
		t.Fatalf("out=%q service=%q", out, *service)
	}
}

func TestPlayMusicDefaultsToYouTube(t *testing.T) {
	service := stubSearch(t, nil)

	out := call(t, "play_music", &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}, Args{"query": "zzz"})

	if out != "No results found for 'zzz'." || *service != "youtube" {
		t.Fatalf("out=%q service=%q", out, *service)
	}
}

func TestPlayMusicGroupAndControls(t *testing.T) {
	stubSearch(t, []music.MusicTrack{{Title: "Song", Artist: "Band"}})
	group := music.NewMusicGroup(48000, 2, nil)
	defer group.Close(context.Background())
	actx := &Context{MusicGroup: group}

	play := call(t, "play_music", actx, Args{"query": "song"})
	now := call(t, "now_playing", actx, Args{})
	pause := call(t, "pause_music", actx, Args{})
	resume := call(t, "resume_music", actx, Args{})
	skip := call(t, "skip_track", actx, Args{})
	stop := call(t, "stop_music", actx, Args{})

	if play != "Playing 'Song' by Band on all devices." || now != "Currently playing: 'Song' by Band (track 1 of 1)." {
		t.Fatalf("play=%q now=%q", play, now)
	}
	if pause != "Paused 'Song' by Band." || resume != "Resuming 'Song' by Band." || skip != "No more tracks in the queue." || stop != "Music stopped." {
		t.Fatalf("pause=%q resume=%q skip=%q stop=%q", pause, resume, skip, stop)
	}
}

func TestMusicControlsWithoutQueue(t *testing.T) {
	stubSearch(t, []music.MusicTrack{{Title: "S", Artist: "A"}})
	actx := &Context{MusicPlayer: music.NewMusicPlayer(48000, 2, nil)}

	for name, want := range map[string]string{
		"pause_music": "No music is playing.", "resume_music": "No music to resume.",
		"skip_track": "No music is playing.", "stop_music": "No music is playing.", "now_playing": "No music is playing.",
	} {
		if got := call(t, name, actx, Args{}); got != want {
			t.Errorf("%s = %q", name, got)
		}
	}
	if got := call(t, "play_music", &Context{}, Args{"query": "s"}); got != "Music playback is not available on this device." {
		t.Fatalf("got %q", got)
	}
}

func TestAutomationTools(t *testing.T) {
	actx := &Context{Scheduler: NewScheduler(filepath.Join(t.TempDir(), "a.json"), nil, nil)}

	created := call(t, "create_automation", actx, Args{"name": "morning", "schedule": "0 7 * * *", "prompt": "weather?"})
	invalid := call(t, "create_automation", actx, Args{"name": "bad", "schedule": "nope", "prompt": "x"})
	list := call(t, "list_automations", actx, Args{})
	toggled := call(t, "toggle_automation", actx, Args{"name": "morning", "enabled": false})
	deleted := call(t, "delete_automation", actx, Args{"name": "morning"})
	missing := call(t, "delete_automation", actx, Args{"name": "morning"})

	if created != "Automation 'morning' created. Schedule: 0 7 * * *." || !strings.HasPrefix(invalid, "invalid cron") {
		t.Fatalf("created=%q invalid=%q", created, invalid)
	}
	if !strings.Contains(list, "'morning' (enabled)") || toggled != "Automation 'morning' disabled." ||
		deleted != "Automation 'morning' deleted." || missing != "No automation named 'morning'." {
		t.Fatalf("list=%q toggled=%q deleted=%q missing=%q", list, toggled, deleted, missing)
	}
	if call(t, "list_automations", &Context{}, Args{}) != "Automations are not available." {
		t.Fatal("expected unavailable")
	}
}
