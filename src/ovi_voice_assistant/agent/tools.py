"""Built-in tools for the voice assistant."""

import ast
import datetime
import math
import random

from agents import RunContextWrapper, function_tool

from ovi_voice_assistant.agent.assistant_context import AssistantContext


@function_tool(is_enabled=False)
async def say(ctx: RunContextWrapper[AssistantContext], text: str) -> str:
    """Say something to the user right now, then continue working.

    Use this to acknowledge a request before performing a slow tool call,
    so the user isn't left in silence. Do not repeat what you say here
    in your final response — give the real answer there instead.

    Args:
        text: The message to speak immediately.
    """
    print(f"[SAY] {text}")  # For debugging and testing without TTS
    if ctx.context.say:
        await ctx.context.say(text)
        return "Spoken."
    return "Say not available."


@function_tool
def get_current_time(timezone: str | None = None) -> str:
    """Get the current date and time.

    Args:
        timezone: IANA timezone name (e.g. 'America/New_York'). Defaults to local time.
    """
    from zoneinfo import ZoneInfo

    if timezone:
        now = datetime.datetime.now(ZoneInfo(timezone))
    else:
        now = datetime.datetime.now().astimezone()

    return now.strftime("%A, %B %d, %Y at %I:%M %p %Z")


@function_tool
async def set_timer(
    ctx: RunContextWrapper[AssistantContext],
    minutes: float = 0,
    seconds: float = 0,
    label: str = "timer",
) -> str:
    """Set a countdown timer. The device will announce when it expires.

    Args:
        minutes: Duration in minutes.
        seconds: Duration in seconds (added to minutes).
        label: A short label for the timer (e.g. 'pasta', 'laundry').
    """
    total_seconds = int(minutes * 60 + seconds)
    if total_seconds <= 0:
        return "Timer duration must be greater than zero."
    ctx.context.schedule_timer(total_seconds, label)

    m, s = divmod(total_seconds, 60)
    h, m = divmod(m, 60)
    parts = []
    if h:
        parts.append(f"{h} hour{'s' if h != 1 else ''}")
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if s:
        parts.append(f"{s} second{'s' if s != 1 else ''}")
    return f"Timer '{label}' set for {', '.join(parts)}."


@function_tool
async def check_timer(ctx: RunContextWrapper[AssistantContext]) -> str:
    """Check the status of all active timers."""
    status = ctx.context.get_timer_status()
    if not status:
        return "No active timers."
    parts = []
    for label, remaining in status.items():
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        time_parts = []
        if h:
            time_parts.append(f"{h}h")
        if m:
            time_parts.append(f"{m}m")
        time_parts.append(f"{s}s")
        parts.append(f"'{label}': {' '.join(time_parts)} remaining")
    return "; ".join(parts)


@function_tool
async def cancel_timer(
    ctx: RunContextWrapper[AssistantContext], label: str = "timer"
) -> str:
    """Cancel an active timer.

    Args:
        label: The label of the timer to cancel.
    """
    if ctx.context.cancel_timer(label):
        return f"Timer '{label}' cancelled."
    return f"No active timer named '{label}'."


@function_tool
def calculate(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e.

    Args:
        expression: The math expression to evaluate (e.g. '2 ** 10', 'sqrt(144)', 'sin(pi/2)').
    """
    allowed = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
    }
    try:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Attribute, ast.Subscript)):
                return "Error: attribute access and indexing are not allowed"
        result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, allowed)
        if isinstance(result, float) and result == int(result):
            return str(int(result))
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@function_tool
def roll_dice(sides: int = 6, count: int = 1) -> str:
    """Roll dice.

    Args:
        sides: Number of sides per die.
        count: Number of dice to roll.
    """
    rolls = [random.randint(1, sides) for _ in range(count)]
    if count == 1:
        return str(rolls[0])
    return f"{rolls} (total: {sum(rolls)})"


@function_tool
def random_number(low: int = 1, high: int = 100) -> str:
    """Pick a random number in a range.

    Args:
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).
    """
    return str(random.randint(low, high))


@function_tool
def flip_coin() -> str:
    """Flip a coin."""
    return random.choice(["Heads", "Tails"])


@function_tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.

    Args:
        value: The numeric value to convert.
        from_unit: Source unit (e.g. 'km', 'miles', 'celsius', 'fahrenheit', 'kg', 'lbs').
        to_unit: Target unit.
    """
    conversions: dict[tuple[str, str], float] = {
        # Length
        ("km", "miles"): 0.621371,
        ("miles", "km"): 1.60934,
        ("m", "ft"): 3.28084,
        ("ft", "m"): 0.3048,
        ("cm", "in"): 0.393701,
        ("in", "cm"): 2.54,
        ("m", "km"): 0.001,
        ("km", "m"): 1000,
        ("ft", "miles"): 1 / 5280,
        ("miles", "ft"): 5280,
        # Weight
        ("kg", "lbs"): 2.20462,
        ("lbs", "kg"): 0.453592,
        ("g", "oz"): 0.035274,
        ("oz", "g"): 28.3495,
        # Volume
        ("liters", "gallons"): 0.264172,
        ("gallons", "liters"): 3.78541,
        ("ml", "oz"): 0.033814,
        ("oz", "ml"): 29.5735,
        # Speed
        ("km/h", "mph"): 0.621371,
        ("mph", "km/h"): 1.60934,
    }

    f, t = from_unit.lower(), to_unit.lower()

    # Temperature special cases
    if f in ("celsius", "c") and t in ("fahrenheit", "f"):
        return f"{round(value * 9 / 5 + 32, 2)} {to_unit}"
    if f in ("fahrenheit", "f") and t in ("celsius", "c"):
        return f"{round((value - 32) * 5 / 9, 2)} {to_unit}"

    factor = conversions.get((f, t))
    if factor is None:
        return f"Unknown conversion: {from_unit} to {to_unit}"
    return f"{round(value * factor, 4)} {to_unit}"


@function_tool
async def play_music(ctx: RunContextWrapper[AssistantContext], query: str) -> str:
    """Search for and play music. Plays on all devices in sync when multiple are connected.

    Args:
        query: What to play — a song name, artist, genre, album, or playlist description.
    """
    from ovi_voice_assistant.music import search_music

    try:
        tracks = await search_music(query)
    except Exception as exc:
        return f"Music search failed: {exc}"

    if not tracks:
        return f"No results found for '{query}'."

    track = tracks[0]
    extra = f" and {len(tracks) - 1} more tracks" if len(tracks) > 1 else ""

    # Use group (multi-device sync) when available, else per-device player
    group = ctx.context.music_group
    if group is not None:
        await group.play(tracks)
        return f"Playing '{track.title}' by {track.artist}{extra} on all devices."

    player = ctx.context.music_player
    if player is None:
        return "Music playback is not available on this device."
    player.set_queue(tracks)
    return f"Playing '{track.title}' by {track.artist}{extra}."


@function_tool
async def pause_music(ctx: RunContextWrapper[AssistantContext]) -> str:
    """Pause the currently playing music on all devices."""
    group = ctx.context.music_group
    if group is not None and group.player.queue:
        await group.pause()
        track = group.player.get_current()
        if track:
            return f"Paused '{track.title}' by {track.artist}."
        return "Music paused."

    player = ctx.context.music_player
    if player is None or not player.queue:
        return "No music is playing."
    player.pause()
    track = player.get_current()
    if track:
        return f"Paused '{track.title}' by {track.artist}."
    return "Music paused."


@function_tool
async def resume_music(ctx: RunContextWrapper[AssistantContext]) -> str:
    """Resume paused music on all devices."""
    group = ctx.context.music_group
    if group is not None and group.player.queue:
        await group.resume()
        track = group.player.get_current()
        if track:
            return f"Resuming '{track.title}' by {track.artist}."
        return "Resuming music."

    player = ctx.context.music_player
    if player is None or not player.queue:
        return "No music to resume."
    player.resume()
    track = player.get_current()
    if track:
        return f"Resuming '{track.title}' by {track.artist}."
    return "Resuming music."


@function_tool
async def skip_track(ctx: RunContextWrapper[AssistantContext]) -> str:
    """Skip to the next track on all devices."""
    group = ctx.context.music_group
    if group is not None and group.player.queue:
        track = group.skip()
        if track:
            return f"Skipping to '{track.title}' by {track.artist}."
        return "No more tracks in the queue."

    player = ctx.context.music_player
    if player is None or not player.queue:
        return "No music is playing."
    track = player.skip()
    if track:
        return f"Skipping to '{track.title}' by {track.artist}."
    return "No more tracks in the queue."


@function_tool
async def stop_music(ctx: RunContextWrapper[AssistantContext]) -> str:
    """Stop music playback and clear the queue on all devices."""
    group = ctx.context.music_group
    if group is not None and group.player.queue:
        await group.stop()
        return "Music stopped."

    player = ctx.context.music_player
    if player is None or not player.queue:
        return "No music is playing."
    player.stop()
    return "Music stopped."


@function_tool
async def now_playing(ctx: RunContextWrapper[AssistantContext]) -> str:
    """Check what music is currently playing or queued."""
    group = ctx.context.music_group
    player = (
        group.player
        if group is not None and group.player.queue
        else ctx.context.music_player
    )
    if player is None or not player.queue:
        return "No music is playing."
    track = player.get_current()
    if not track:
        return "No music is playing."
    pos = player.current_index + 1
    total = len(player.queue)
    status = "playing" if player.is_active else "paused"
    return f"Currently {status}: '{track.title}' by {track.artist} (track {pos} of {total})."


@function_tool
async def create_automation(
    ctx: RunContextWrapper[AssistantContext],
    name: str,
    schedule: str,
    prompt: str,
) -> str:
    """Create a recurring automation that runs on a schedule and announces the result.

    The automation runs the prompt through the AI agent at the scheduled time
    and speaks the response on the device. Use this when the user says things
    like "every morning at 7, tell me the weather" or "remind me to stretch
    every hour".

    Args:
        name: Short label for the automation (e.g. 'morning weather').
        schedule: Cron expression with 5 fields: minute hour day-of-month month day-of-week.
                  Examples: '0 7 * * *' = daily 7 AM, '0 7 * * 1-5' = weekdays 7 AM,
                  '*/30 * * * *' = every 30 minutes, '0 18 * * 5' = Fridays 6 PM.
                  Day-of-week: 0=Sunday, 1=Monday, ..., 6=Saturday.
        prompt: What to ask the agent when it fires (e.g. 'What is the weather forecast for today?').
    """
    scheduler = ctx.context.scheduler
    if scheduler is None:
        return "Automations are not available."
    try:
        auto = scheduler.create(name, schedule, prompt)
        return f"Automation '{auto.name}' created. Schedule: {auto.schedule}."
    except ValueError as exc:
        return str(exc)


@function_tool
async def list_automations(ctx: RunContextWrapper[AssistantContext]) -> str:
    """List all scheduled automations and their status."""
    scheduler = ctx.context.scheduler
    if scheduler is None:
        return "Automations are not available."
    autos = scheduler.automations
    if not autos:
        return "No automations configured."
    parts = []
    for a in autos:
        status = "enabled" if a.enabled else "disabled"
        parts.append(
            f"'{a.name}' ({status}): schedule={a.schedule}, prompt={a.prompt!r}"
        )
    return "; ".join(parts)


@function_tool
async def delete_automation(ctx: RunContextWrapper[AssistantContext], name: str) -> str:
    """Delete a scheduled automation.

    Args:
        name: The name of the automation to delete.
    """
    scheduler = ctx.context.scheduler
    if scheduler is None:
        return "Automations are not available."
    if scheduler.delete(name):
        return f"Automation '{name}' deleted."
    return f"No automation named '{name}'."


@function_tool
async def toggle_automation(
    ctx: RunContextWrapper[AssistantContext],
    name: str,
    enabled: bool,
) -> str:
    """Enable or disable a scheduled automation without deleting it.

    Args:
        name: The name of the automation to toggle.
        enabled: True to enable, False to disable.
    """
    scheduler = ctx.context.scheduler
    if scheduler is None:
        return "Automations are not available."
    if scheduler.set_enabled(name, enabled):
        state = "enabled" if enabled else "disabled"
        return f"Automation '{name}' {state}."
    return f"No automation named '{name}'."


BUILTIN_TOOLS = [
    say,
    get_current_time,
    set_timer,
    check_timer,
    cancel_timer,
    calculate,
    roll_dice,
    random_number,
    flip_coin,
    unit_convert,
    play_music,
    pause_music,
    resume_music,
    skip_track,
    stop_music,
    now_playing,
    create_automation,
    list_automations,
    delete_automation,
    toggle_automation,
]
