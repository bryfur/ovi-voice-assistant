package agent

import (
	"context"
	"fmt"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// builtinTools returns the built-in tools for the voice assistant.
func builtinTools() []Tool {
	return []Tool{
		{
			Name:        "get_current_time",
			Description: "Get the current date and time.",
			Parameters: schema(nil, map[string]any{
				"timezone": prop("string", "IANA timezone name (e.g. 'America/New_York'). Defaults to local time."),
			}),
			Handler: toolGetCurrentTime,
		},
		{
			Name:        "set_timer",
			Description: "Set a countdown timer. The device will announce when it expires.",
			Parameters: schema(nil, map[string]any{
				"minutes": prop("number", "Duration in minutes."),
				"seconds": prop("number", "Duration in seconds (added to minutes)."),
				"label":   prop("string", "A short label for the timer (e.g. 'pasta', 'laundry')."),
			}),
			Handler: toolSetTimer,
		},
		{
			Name:        "check_timer",
			Description: "Check the status of all active timers.",
			Handler:     toolCheckTimer,
		},
		{
			Name:        "cancel_timer",
			Description: "Cancel an active timer.",
			Parameters: schema(nil, map[string]any{
				"label": prop("string", "The label of the timer to cancel."),
			}),
			Handler: toolCancelTimer,
		},
		{
			Name:        "calculate",
			Description: "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.",
			Parameters: schema([]string{"expression"}, map[string]any{
				"expression": prop("string", "The math expression to evaluate (e.g. '2 ** 10', 'sqrt(144)', 'sin(pi/2)')."),
			}),
			Handler: toolCalculate,
		},
		{
			Name:        "roll_dice",
			Description: "Roll dice.",
			Parameters: schema(nil, map[string]any{
				"sides": prop("integer", "Number of sides per die."),
				"count": prop("integer", "Number of dice to roll."),
			}),
			Handler: toolRollDice,
		},
		{
			Name:        "random_number",
			Description: "Pick a random number in a range.",
			Parameters: schema(nil, map[string]any{
				"low":  prop("integer", "Lower bound (inclusive)."),
				"high": prop("integer", "Upper bound (inclusive)."),
			}),
			Handler: toolRandomNumber,
		},
		{
			Name:        "flip_coin",
			Description: "Flip a coin.",
			Handler:     toolFlipCoin,
		},
		{
			Name:        "unit_convert",
			Description: "Convert between common units.",
			Parameters: schema([]string{"value", "from_unit", "to_unit"}, map[string]any{
				"value":     prop("number", "The numeric value to convert."),
				"from_unit": prop("string", "Source unit (e.g. 'km', 'miles', 'celsius', 'fahrenheit', 'kg', 'lbs')."),
				"to_unit":   prop("string", "Target unit."),
			}),
			Handler: toolUnitConvert,
		},
		{
			Name:        "play_music",
			Description: "Search for and play music. Plays on all devices in sync when multiple are connected.",
			Parameters: schema([]string{"query"}, map[string]any{
				"query":   prop("string", "What to play — a song name, artist, genre, album, or playlist description."),
				"service": prop("string", "Music service to use: youtube (default), spotify or apple. Only services the user enabled work."),
			}),
			Handler: toolPlayMusic,
		},
		{Name: "pause_music", Description: "Pause the currently playing music on all devices.", Handler: toolPauseMusic},
		{Name: "resume_music", Description: "Resume paused music on all devices.", Handler: toolResumeMusic},
		{Name: "skip_track", Description: "Skip to the next track on all devices.", Handler: toolSkipTrack},
		{Name: "stop_music", Description: "Stop music playback and clear the queue on all devices.", Handler: toolStopMusic},
		{Name: "now_playing", Description: "Check what music is currently playing or queued.", Handler: toolNowPlaying},
		{
			Name: "create_automation",
			Description: "Create a recurring automation that runs on a schedule and announces the result.\n\n" +
				"The automation runs the prompt through the AI agent at the scheduled time " +
				"and speaks the response on the device. Use this when the user says things " +
				"like \"every morning at 7, tell me the weather\" or \"remind me to stretch " +
				"every hour\".",
			Parameters: schema([]string{"name", "schedule", "prompt"}, map[string]any{
				"name": prop("string", "Short label for the automation (e.g. 'morning weather')."),
				"schedule": prop("string", "Cron expression with 5 fields: minute hour day-of-month month day-of-week. "+
					"Examples: '0 7 * * *' = daily 7 AM, '0 7 * * 1-5' = weekdays 7 AM, "+
					"'*/30 * * * *' = every 30 minutes, '0 18 * * 5' = Fridays 6 PM. "+
					"Day-of-week: 0=Sunday, 1=Monday, ..., 6=Saturday."),
				"prompt": prop("string", "What to ask the agent when it fires (e.g. 'What is the weather forecast for today?')."),
			}),
			Handler: toolCreateAutomation,
		},
		{Name: "list_automations", Description: "List all scheduled automations and their status.", Handler: toolListAutomations},
		{
			Name:        "delete_automation",
			Description: "Delete a scheduled automation.",
			Parameters: schema([]string{"name"}, map[string]any{
				"name": prop("string", "The name of the automation to delete."),
			}),
			Handler: toolDeleteAutomation,
		},
		{
			Name:        "toggle_automation",
			Description: "Enable or disable a scheduled automation without deleting it.",
			Parameters: schema([]string{"name", "enabled"}, map[string]any{
				"name":    prop("string", "The name of the automation to toggle."),
				"enabled": prop("boolean", "True to enable, False to disable."),
			}),
			Handler: toolToggleAutomation,
		},
	}
}

// -- Time --

func toolGetCurrentTime(_ context.Context, _ *Context, args Args) (string, error) {
	now := time.Now()
	if tz := args.String("timezone", ""); tz != "" {
		loc, err := time.LoadLocation(tz)
		if err != nil {
			return "", fmt.Errorf("unknown timezone %q", tz)
		}
		now = now.In(loc)
	}
	return now.Format("Monday, January 02, 2006 at 03:04 PM MST"), nil
}

// -- Timers --

func plural(n int, unit string) string {
	if n == 1 {
		return fmt.Sprintf("%d %s", n, unit)
	}
	return fmt.Sprintf("%d %ss", n, unit)
}

func toolSetTimer(_ context.Context, actx *Context, args Args) (string, error) {
	minutes := args.Float("minutes", 0)
	seconds := args.Float("seconds", 0)
	label := args.String("label", "timer")
	total := int(minutes*60 + seconds)
	if total <= 0 {
		return "Timer duration must be greater than zero.", nil
	}
	actx.ScheduleTimer(time.Duration(total)*time.Second, label)
	h := total / 3600
	m := (total % 3600) / 60
	s := total % 60
	var parts []string
	if h > 0 {
		parts = append(parts, plural(h, "hour"))
	}
	if m > 0 {
		parts = append(parts, plural(m, "minute"))
	}
	if s > 0 {
		parts = append(parts, plural(s, "second"))
	}
	return fmt.Sprintf("Timer '%s' set for %s.", label, strings.Join(parts, ", ")), nil
}

func toolCheckTimer(_ context.Context, actx *Context, _ Args) (string, error) {
	status := actx.TimerStatus()
	if len(status) == 0 {
		return "No active timers.", nil
	}
	labels := make([]string, 0, len(status))
	for l := range status {
		labels = append(labels, l)
	}
	sort.Strings(labels)
	var parts []string
	for _, label := range labels {
		remaining := int(status[label].Seconds())
		h := remaining / 3600
		m := (remaining % 3600) / 60
		s := remaining % 60
		var tp []string
		if h > 0 {
			tp = append(tp, fmt.Sprintf("%dh", h))
		}
		if m > 0 {
			tp = append(tp, fmt.Sprintf("%dm", m))
		}
		tp = append(tp, fmt.Sprintf("%ds", s))
		parts = append(parts, fmt.Sprintf("'%s': %s remaining", label, strings.Join(tp, " ")))
	}
	return strings.Join(parts, "; "), nil
}

func toolCancelTimer(_ context.Context, actx *Context, args Args) (string, error) {
	label := args.String("label", "timer")
	if actx.CancelTimer(label) {
		return fmt.Sprintf("Timer '%s' cancelled.", label), nil
	}
	return fmt.Sprintf("No active timer named '%s'.", label), nil
}

// -- Math --

func toolCalculate(_ context.Context, _ *Context, args Args) (string, error) {
	result, err := calculate(args.String("expression", ""))
	if err != nil {
		return "Error: " + err.Error(), nil
	}
	return result, nil
}

// -- Random --

func toolRollDice(_ context.Context, _ *Context, args Args) (string, error) {
	sides := args.Int("sides", 6)
	count := args.Int("count", 1)
	if sides < 1 {
		sides = 1
	}
	if count < 1 {
		count = 1
	}
	rolls := make([]int, count)
	total := 0
	for i := range rolls {
		rolls[i] = rand.IntN(sides) + 1
		total += rolls[i]
	}
	if count == 1 {
		return strconv.Itoa(rolls[0]), nil
	}
	strs := make([]string, count)
	for i, r := range rolls {
		strs[i] = strconv.Itoa(r)
	}
	return fmt.Sprintf("[%s] (total: %d)", strings.Join(strs, ", "), total), nil
}

func toolRandomNumber(_ context.Context, _ *Context, args Args) (string, error) {
	low := args.Int("low", 1)
	high := args.Int("high", 100)
	if high < low {
		low, high = high, low
	}
	return strconv.Itoa(low + rand.IntN(high-low+1)), nil
}

func toolFlipCoin(_ context.Context, _ *Context, _ Args) (string, error) {
	if rand.IntN(2) == 0 {
		return "Heads", nil
	}
	return "Tails", nil
}

// -- Units --

type unitPair struct{ from, to string }

var unitConversions = map[unitPair]float64{
	// Length
	{"km", "miles"}: 0.621371,
	{"miles", "km"}: 1.60934,
	{"m", "ft"}:     3.28084,
	{"ft", "m"}:     0.3048,
	{"cm", "in"}:    0.393701,
	{"in", "cm"}:    2.54,
	{"m", "km"}:     0.001,
	{"km", "m"}:     1000,
	{"ft", "miles"}: 1.0 / 5280,
	{"miles", "ft"}: 5280,
	// Weight
	{"kg", "lbs"}: 2.20462,
	{"lbs", "kg"}: 0.453592,
	{"g", "oz"}:   0.035274,
	{"oz", "g"}:   28.3495,
	// Volume
	{"liters", "gallons"}: 0.264172,
	{"gallons", "liters"}: 3.78541,
	{"ml", "oz"}:          0.033814,
	{"oz", "ml"}:          29.5735,
	// Speed
	{"km/h", "mph"}: 0.621371,
	{"mph", "km/h"}: 1.60934,
}

func roundTo(v float64, places int) string {
	s := strconv.FormatFloat(v, 'f', places, 64)
	if strings.Contains(s, ".") {
		s = strings.TrimRight(strings.TrimRight(s, "0"), ".")
	}
	return s
}

func toolUnitConvert(_ context.Context, _ *Context, args Args) (string, error) {
	value := args.Float("value", 0)
	fromUnit := args.String("from_unit", "")
	toUnit := args.String("to_unit", "")
	f, t := strings.ToLower(fromUnit), strings.ToLower(toUnit)

	isC := func(u string) bool { return u == "celsius" || u == "c" }
	isF := func(u string) bool { return u == "fahrenheit" || u == "f" }
	if isC(f) && isF(t) {
		return fmt.Sprintf("%s %s", roundTo(value*9/5+32, 2), toUnit), nil
	}
	if isF(f) && isC(t) {
		return fmt.Sprintf("%s %s", roundTo((value-32)*5/9, 2), toUnit), nil
	}
	factor, ok := unitConversions[unitPair{f, t}]
	if !ok {
		return fmt.Sprintf("Unknown conversion: %s to %s", fromUnit, toUnit), nil
	}
	return fmt.Sprintf("%s %s", roundTo(value*factor, 4), toUnit), nil
}

// -- Music --

func toolPlayMusic(ctx context.Context, actx *Context, args Args) (string, error) {
	query := args.String("query", "")
	tracks, err := music.SearchMusicFunc(ctx, query, args.String("service", "youtube"))
	if err != nil {
		return fmt.Sprintf("Music search failed: %v", err), nil
	}
	if len(tracks) == 0 {
		return fmt.Sprintf("No results found for '%s'.", query), nil
	}
	track := tracks[0]
	extra := ""
	if len(tracks) > 1 {
		extra = fmt.Sprintf(" and %d more tracks", len(tracks)-1)
	}
	// Use group (multi-device sync) when available, else per-device player
	if actx.MusicGroup != nil {
		actx.MusicGroup.Play(ctx, tracks)
		return fmt.Sprintf("Playing '%s' by %s%s on all devices.", track.Title, track.Artist, extra), nil
	}
	if actx.MusicPlayer == nil {
		return "Music playback is not available on this device.", nil
	}
	actx.MusicPlayer.SetQueue(tracks, 0)
	return fmt.Sprintf("Playing '%s' by %s%s.", track.Title, track.Artist, extra), nil
}

func groupHasQueue(actx *Context) bool {
	return actx.MusicGroup != nil && actx.MusicGroup.Player.QueueLen() > 0
}

func toolPauseMusic(ctx context.Context, actx *Context, _ Args) (string, error) {
	if groupHasQueue(actx) {
		actx.MusicGroup.Pause(ctx)
		if t := actx.MusicGroup.Player.GetCurrent(); t != nil {
			return fmt.Sprintf("Paused '%s' by %s.", t.Title, t.Artist), nil
		}
		return "Music paused.", nil
	}
	p := actx.MusicPlayer
	if p == nil || p.QueueLen() == 0 {
		return "No music is playing.", nil
	}
	p.Pause()
	if t := p.GetCurrent(); t != nil {
		return fmt.Sprintf("Paused '%s' by %s.", t.Title, t.Artist), nil
	}
	return "Music paused.", nil
}

func toolResumeMusic(ctx context.Context, actx *Context, _ Args) (string, error) {
	if groupHasQueue(actx) {
		actx.MusicGroup.Resume(ctx)
		if t := actx.MusicGroup.Player.GetCurrent(); t != nil {
			return fmt.Sprintf("Resuming '%s' by %s.", t.Title, t.Artist), nil
		}
		return "Resuming music.", nil
	}
	p := actx.MusicPlayer
	if p == nil || p.QueueLen() == 0 {
		return "No music to resume.", nil
	}
	p.Resume()
	if t := p.GetCurrent(); t != nil {
		return fmt.Sprintf("Resuming '%s' by %s.", t.Title, t.Artist), nil
	}
	return "Resuming music.", nil
}

func toolSkipTrack(_ context.Context, actx *Context, _ Args) (string, error) {
	if groupHasQueue(actx) {
		if t := actx.MusicGroup.Skip(); t != nil {
			return fmt.Sprintf("Skipping to '%s' by %s.", t.Title, t.Artist), nil
		}
		return "No more tracks in the queue.", nil
	}
	p := actx.MusicPlayer
	if p == nil || p.QueueLen() == 0 {
		return "No music is playing.", nil
	}
	if t := p.Skip(); t != nil {
		return fmt.Sprintf("Skipping to '%s' by %s.", t.Title, t.Artist), nil
	}
	return "No more tracks in the queue.", nil
}

func toolStopMusic(ctx context.Context, actx *Context, _ Args) (string, error) {
	if groupHasQueue(actx) {
		actx.MusicGroup.Stop(ctx)
		return "Music stopped.", nil
	}
	p := actx.MusicPlayer
	if p == nil || p.QueueLen() == 0 {
		return "No music is playing.", nil
	}
	p.Stop()
	return "Music stopped.", nil
}

func toolNowPlaying(_ context.Context, actx *Context, _ Args) (string, error) {
	var p *music.MusicPlayer
	if groupHasQueue(actx) {
		p = actx.MusicGroup.Player
	} else {
		p = actx.MusicPlayer
	}
	if p == nil || p.QueueLen() == 0 {
		return "No music is playing.", nil
	}
	t := p.GetCurrent()
	if t == nil {
		return "No music is playing.", nil
	}
	status := "paused"
	if p.IsActive() {
		status = "playing"
	}
	return fmt.Sprintf("Currently %s: '%s' by %s (track %d of %d).",
		status, t.Title, t.Artist, p.CurrentIndex()+1, p.QueueLen()), nil
}

// -- Automations --

func toolCreateAutomation(_ context.Context, actx *Context, args Args) (string, error) {
	if actx.Scheduler == nil {
		return "Automations are not available.", nil
	}
	auto, err := actx.Scheduler.Create(args.String("name", ""), args.String("schedule", ""), args.String("prompt", ""))
	if err != nil {
		return err.Error(), nil
	}
	return fmt.Sprintf("Automation '%s' created. Schedule: %s.", auto.Name, auto.Schedule), nil
}

func toolListAutomations(_ context.Context, actx *Context, _ Args) (string, error) {
	if actx.Scheduler == nil {
		return "Automations are not available.", nil
	}
	autos := actx.Scheduler.Automations()
	if len(autos) == 0 {
		return "No automations configured.", nil
	}
	parts := make([]string, 0, len(autos))
	for _, a := range autos {
		status := "disabled"
		if a.Enabled {
			status = "enabled"
		}
		parts = append(parts, fmt.Sprintf("'%s' (%s): schedule=%s, prompt=%q", a.Name, status, a.Schedule, a.Prompt))
	}
	return strings.Join(parts, "; "), nil
}

func toolDeleteAutomation(_ context.Context, actx *Context, args Args) (string, error) {
	if actx.Scheduler == nil {
		return "Automations are not available.", nil
	}
	name := args.String("name", "")
	if actx.Scheduler.Delete(name) {
		return fmt.Sprintf("Automation '%s' deleted.", name), nil
	}
	return fmt.Sprintf("No automation named '%s'.", name), nil
}

func toolToggleAutomation(_ context.Context, actx *Context, args Args) (string, error) {
	if actx.Scheduler == nil {
		return "Automations are not available.", nil
	}
	name := args.String("name", "")
	enabled := args.Bool("enabled", true)
	if actx.Scheduler.SetEnabled(name, enabled) {
		state := "disabled"
		if enabled {
			state = "enabled"
		}
		return fmt.Sprintf("Automation '%s' %s.", name, state), nil
	}
	return fmt.Sprintf("No automation named '%s'.", name), nil
}
