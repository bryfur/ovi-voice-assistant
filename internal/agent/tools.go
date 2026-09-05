package agent

import (
	"context"
	"fmt"
	"maps"
	"math"
	"math/rand/v2"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/bryfur/ovi-voice-assistant/internal/music"
)

// builtinTools are always available to the model.
func builtinTools() []Tool {
	return []Tool{
		{
			Name:        "get_current_time",
			Description: "Get the current date and time.",
			Parameters:  schema(nil, map[string]any{"timezone": prop("string", "IANA timezone name (e.g. 'America/New_York'). Defaults to local time.")}),
			Run:         currentTime,
		},
		{
			Name:        "set_timer",
			Description: "Set a countdown timer. The device will announce when it expires.",
			Parameters: schema(nil, map[string]any{
				"minutes": prop("number", "Duration in minutes."),
				"seconds": prop("number", "Duration in seconds (added to minutes)."),
				"label":   prop("string", "A short label for the timer (e.g. 'pasta', 'laundry')."),
			}),
			Run: setTimer,
		},
		{Name: "check_timer", Description: "Check the status of all active timers.", Run: checkTimers},
		{
			Name:        "cancel_timer",
			Description: "Cancel an active timer.",
			Parameters:  schema(nil, map[string]any{"label": prop("string", "The label of the timer to cancel.")}),
			Run:         cancelTimer,
		},
		{
			Name:        "calculate",
			Description: "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.",
			Parameters:  schema([]string{"expression"}, map[string]any{"expression": prop("string", "The math expression to evaluate (e.g. '2 ** 10', 'sqrt(144)', 'sin(pi/2)').")}),
			Run: func(_ context.Context, _ *Env, a Args) (string, error) {
				out, err := calculate(a.String("expression", ""))
				if err != nil {
					return "Error: " + err.Error(), nil
				}
				return out, nil
			},
		},
		{
			Name:        "roll_dice",
			Description: "Roll dice.",
			Parameters: schema(nil, map[string]any{
				"sides": prop("integer", "Number of sides per die."),
				"count": prop("integer", "Number of dice to roll."),
			}),
			Run: rollDice,
		},
		{
			Name:        "random_number",
			Description: "Pick a random number in a range.",
			Parameters: schema(nil, map[string]any{
				"low":  prop("integer", "Lower bound (inclusive)."),
				"high": prop("integer", "Upper bound (inclusive)."),
			}),
			Run: func(_ context.Context, _ *Env, a Args) (string, error) {
				lo, hi := a.Int("low", 1), a.Int("high", 100)
				lo, hi = min(lo, hi), max(lo, hi)
				return strconv.Itoa(lo + rand.IntN(hi-lo+1)), nil
			},
		},
		{
			Name: "flip_coin", Description: "Flip a coin.",
			Run: func(context.Context, *Env, Args) (string, error) {
				return []string{"Heads", "Tails"}[rand.IntN(2)], nil
			},
		},
		{
			Name:        "unit_convert",
			Description: "Convert between common units.",
			Parameters: schema([]string{"value", "from_unit", "to_unit"}, map[string]any{
				"value":     prop("number", "The numeric value to convert."),
				"from_unit": prop("string", "Source unit (e.g. 'km', 'miles', 'celsius', 'fahrenheit', 'kg', 'lbs')."),
				"to_unit":   prop("string", "Target unit."),
			}),
			Run: convertUnits,
		},
		{
			Name:        "play_music",
			Description: "Search for and play music. Plays on all devices in sync when multiple are connected.",
			Parameters: schema([]string{"query"}, map[string]any{
				"query":   prop("string", "What to play: a song name, artist, genre, album, or playlist description."),
				"service": prop("string", "Music service to use: youtube (default), spotify or apple. Only services the user enabled work."),
			}),
			Run: playMusic,
		},
		{Name: "pause_music", Description: "Pause the currently playing music on all devices.", Run: pauseMusic},
		{Name: "resume_music", Description: "Resume paused music on all devices.", Run: resumeMusic},
		{Name: "skip_track", Description: "Skip to the next track on all devices.", Run: skipTrack},
		{Name: "stop_music", Description: "Stop music playback and clear the queue on all devices.", Run: stopMusic},
		{Name: "now_playing", Description: "Check what music is currently playing or queued.", Run: nowPlaying},
		{
			Name: "create_automation",
			Description: "Create a recurring automation that runs on a schedule and announces the result. " +
				"The prompt goes through the AI agent at the scheduled time and the response is spoken on the device. " +
				"Use this for requests like \"every morning at 7, tell me the weather\" or \"remind me to stretch every hour\".",
			Parameters: schema([]string{"name", "schedule", "prompt"}, map[string]any{
				"name": prop("string", "Short label for the automation (e.g. 'morning weather')."),
				"schedule": prop("string", "Cron expression with 5 fields: minute hour day-of-month month day-of-week. "+
					"Examples: '0 7 * * *' = daily 7 AM, '0 7 * * 1-5' = weekdays 7 AM, '*/30 * * * *' = every 30 minutes, "+
					"'0 18 * * 5' = Fridays 6 PM. Day-of-week: 0=Sunday, 1=Monday, ..., 6=Saturday."),
				"prompt": prop("string", "What to ask the agent when it fires (e.g. 'What is the weather forecast for today?')."),
			}),
			Run: createAutomation,
		},
		{Name: "list_automations", Description: "List all scheduled automations and their status.", Run: listAutomations},
		{
			Name:        "delete_automation",
			Description: "Delete a scheduled automation.",
			Parameters:  schema([]string{"name"}, map[string]any{"name": prop("string", "The name of the automation to delete.")}),
			Run:         deleteAutomation,
		},
		{
			Name:        "toggle_automation",
			Description: "Enable or disable a scheduled automation without deleting it.",
			Parameters: schema([]string{"name", "enabled"}, map[string]any{
				"name":    prop("string", "The name of the automation to toggle."),
				"enabled": prop("boolean", "True to enable, False to disable."),
			}),
			Run: toggleAutomation,
		},
	}
}

// -- Time and timers --

func currentTime(_ context.Context, _ *Env, a Args) (string, error) {
	now := time.Now()
	if tz := a.String("timezone", ""); tz != "" {
		loc, err := time.LoadLocation(tz)
		if err != nil {
			return "", fmt.Errorf("unknown timezone %q", tz)
		}
		now = now.In(loc)
	}
	return now.Format("Monday, January 02, 2006 at 03:04 PM MST"), nil
}

// spoken renders a duration the way it is said: "2 hours, 5 minutes".
func spoken(d time.Duration) string {
	var parts []string
	for _, u := range []struct {
		name string
		size time.Duration
	}{{"hour", time.Hour}, {"minute", time.Minute}, {"second", time.Second}} {
		if n := int(d / u.size); n > 0 {
			d -= time.Duration(n) * u.size
			parts = append(parts, plural(n, u.name))
		}
	}
	if len(parts) == 0 {
		return "0 seconds"
	}
	return strings.Join(parts, ", ")
}

func plural(n int, unit string) string {
	if n == 1 {
		return "1 " + unit
	}
	return fmt.Sprintf("%d %ss", n, unit)
}

func setTimer(_ context.Context, env *Env, a Args) (string, error) {
	d := time.Duration(a.Float("minutes", 0)*60+a.Float("seconds", 0)) * time.Second
	if d <= 0 {
		return "Timer duration must be greater than zero.", nil
	}
	label := a.String("label", "timer")
	env.SetTimer(d, label)
	return fmt.Sprintf("Timer '%s' set for %s.", label, spoken(d)), nil
}

func checkTimers(_ context.Context, env *Env, _ Args) (string, error) {
	left := env.Timers()
	if len(left) == 0 {
		return "No active timers.", nil
	}
	var parts []string
	for _, label := range slices.Sorted(maps.Keys(left)) {
		parts = append(parts, fmt.Sprintf("'%s': %s remaining", label, spoken(left[label])))
	}
	return strings.Join(parts, "; "), nil
}

func cancelTimer(_ context.Context, env *Env, a Args) (string, error) {
	label := a.String("label", "timer")
	if env.CancelTimer(label) {
		return fmt.Sprintf("Timer '%s' cancelled.", label), nil
	}
	return fmt.Sprintf("No active timer named '%s'.", label), nil
}

// -- Dice and units --

func rollDice(_ context.Context, _ *Env, a Args) (string, error) {
	sides, count := max(a.Int("sides", 6), 1), max(a.Int("count", 1), 1)
	rolls := make([]string, count)
	total := 0
	for i := range rolls {
		r := rand.IntN(sides) + 1
		total += r
		rolls[i] = strconv.Itoa(r)
	}
	if count == 1 {
		return rolls[0], nil
	}
	return fmt.Sprintf("[%s] (total: %d)", strings.Join(rolls, ", "), total), nil
}

// unitFactors multiply a value in the first unit to get the second.
var unitFactors = map[[2]string]float64{
	{"km", "miles"}: 0.621371, {"miles", "km"}: 1.60934,
	{"m", "ft"}: 3.28084, {"ft", "m"}: 0.3048,
	{"cm", "in"}: 0.393701, {"in", "cm"}: 2.54,
	{"m", "km"}: 0.001, {"km", "m"}: 1000,
	{"ft", "miles"}: 1.0 / 5280, {"miles", "ft"}: 5280,
	{"kg", "lbs"}: 2.20462, {"lbs", "kg"}: 0.453592,
	{"g", "oz"}: 0.035274, {"oz", "g"}: 28.3495,
	{"liters", "gallons"}: 0.264172, {"gallons", "liters"}: 3.78541,
	{"ml", "oz"}: 0.033814, {"oz", "ml"}: 29.5735,
	{"km/h", "mph"}: 0.621371, {"mph", "km/h"}: 1.60934,
}

func convertUnits(_ context.Context, _ *Env, a Args) (string, error) {
	v, from, to := a.Float("value", 0), a.String("from_unit", ""), a.String("to_unit", "")
	f, t := strings.ToLower(from), strings.ToLower(to)
	celsius := func(u string) bool { return u == "celsius" || u == "c" }
	fahrenheit := func(u string) bool { return u == "fahrenheit" || u == "f" }
	round := func(x float64, places float64) string {
		p := math.Pow(10, places)
		return strconv.FormatFloat(math.Round(x*p)/p, 'f', -1, 64)
	}
	switch factor, ok := unitFactors[[2]string{f, t}]; {
	case celsius(f) && fahrenheit(t):
		return round(v*9/5+32, 2) + " " + to, nil
	case fahrenheit(f) && celsius(t):
		return round((v-32)*5/9, 2) + " " + to, nil
	case ok:
		return round(v*factor, 4) + " " + to, nil
	}
	return fmt.Sprintf("Unknown conversion: %s to %s", from, to), nil
}

// -- Music --

func describe(t music.Track) string { return fmt.Sprintf("'%s' by %s", t.Title, t.Artist) }

// current is the track playing or paused, nil when there is no music.
func current(env *Env) *music.Track {
	if env.Music == nil {
		return nil
	}
	return env.Music.Current()
}

func playMusic(ctx context.Context, env *Env, a Args) (string, error) {
	if env.Music == nil {
		return "Music playback is not available.", nil
	}
	query := a.String("query", "")
	tracks, err := env.Music.Search(ctx, query, a.String("service", ""))
	if err != nil {
		return "Music search failed: " + err.Error(), nil
	}
	if len(tracks) == 0 {
		return fmt.Sprintf("No results found for '%s'.", query), nil
	}
	env.Music.Play(tracks)
	more := ""
	if n := len(tracks) - 1; n > 0 {
		more = fmt.Sprintf(" and %d more tracks", n)
	}
	return fmt.Sprintf("Playing %s%s.", describe(tracks[0]), more), nil
}

func pauseMusic(_ context.Context, env *Env, _ Args) (string, error) {
	t := current(env)
	if t == nil {
		return "No music is playing.", nil
	}
	env.Music.Pause()
	return fmt.Sprintf("Paused %s.", describe(*t)), nil
}

func resumeMusic(_ context.Context, env *Env, _ Args) (string, error) {
	t := current(env)
	if t == nil {
		return "No music to resume.", nil
	}
	env.Music.Resume()
	return fmt.Sprintf("Resuming %s.", describe(*t)), nil
}

func skipTrack(_ context.Context, env *Env, _ Args) (string, error) {
	if current(env) == nil {
		return "No music is playing.", nil
	}
	if next := env.Music.Skip(); next != nil {
		return fmt.Sprintf("Skipping to %s.", describe(*next)), nil
	}
	return "No more tracks in the queue.", nil
}

func stopMusic(_ context.Context, env *Env, _ Args) (string, error) {
	if current(env) == nil {
		return "No music is playing.", nil
	}
	env.Music.Stop()
	return "Music stopped.", nil
}

func nowPlaying(_ context.Context, env *Env, _ Args) (string, error) {
	t := current(env)
	if t == nil {
		return "No music is playing.", nil
	}
	state := "playing"
	if env.Music.Paused() {
		state = "paused"
	}
	return fmt.Sprintf("Currently %s: %s (%d more queued).", state, describe(*t), env.Music.Remaining()), nil
}

// -- Automations --

const noAutomations = "Automations are not available."

func createAutomation(_ context.Context, env *Env, a Args) (string, error) {
	if env.Scheduler == nil {
		return noAutomations, nil
	}
	auto, err := env.Scheduler.Create(a.String("name", ""), a.String("schedule", ""), a.String("prompt", ""))
	if err != nil {
		return err.Error(), nil
	}
	return fmt.Sprintf("Automation '%s' created. Schedule: %s.", auto.Name, auto.Schedule), nil
}

func listAutomations(_ context.Context, env *Env, _ Args) (string, error) {
	if env.Scheduler == nil {
		return noAutomations, nil
	}
	autos := env.Scheduler.Automations()
	if len(autos) == 0 {
		return "No automations configured.", nil
	}
	parts := make([]string, len(autos))
	for i, auto := range autos {
		state := "disabled"
		if auto.Enabled {
			state = "enabled"
		}
		parts[i] = fmt.Sprintf("'%s' (%s): schedule=%s, prompt=%q", auto.Name, state, auto.Schedule, auto.Prompt)
	}
	return strings.Join(parts, "; "), nil
}

func deleteAutomation(_ context.Context, env *Env, a Args) (string, error) {
	if env.Scheduler == nil {
		return noAutomations, nil
	}
	name := a.String("name", "")
	if env.Scheduler.Delete(name) {
		return fmt.Sprintf("Automation '%s' deleted.", name), nil
	}
	return fmt.Sprintf("No automation named '%s'.", name), nil
}

func toggleAutomation(_ context.Context, env *Env, a Args) (string, error) {
	if env.Scheduler == nil {
		return noAutomations, nil
	}
	name, on := a.String("name", ""), a.Bool("enabled", true)
	if !env.Scheduler.Enable(name, on) {
		return fmt.Sprintf("No automation named '%s'.", name), nil
	}
	if on {
		return fmt.Sprintf("Automation '%s' enabled.", name), nil
	}
	return fmt.Sprintf("Automation '%s' disabled.", name), nil
}
