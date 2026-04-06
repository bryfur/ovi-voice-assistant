"""Built-in tools for the voice assistant."""

import datetime
import math
import random

from agents import RunContextWrapper, function_tool

from open_voice_assistant.agent.assistant_context import AssistantContext


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
async def set_timer(ctx: RunContextWrapper[AssistantContext], minutes: float = 0, seconds: float = 0, label: str = "timer") -> str:
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
async def cancel_timer(ctx: RunContextWrapper[AssistantContext], label: str = "timer") -> str:
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
        "sqrt": math.sqrt, "abs": abs, "round": round,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10, "log2": math.log2,
        "pi": math.pi, "e": math.e,
        "ceil": math.ceil, "floor": math.floor,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
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
        ("km", "miles"): 0.621371, ("miles", "km"): 1.60934,
        ("m", "ft"): 3.28084, ("ft", "m"): 0.3048,
        ("cm", "in"): 0.393701, ("in", "cm"): 2.54,
        ("m", "km"): 0.001, ("km", "m"): 1000,
        ("ft", "miles"): 1 / 5280, ("miles", "ft"): 5280,
        # Weight
        ("kg", "lbs"): 2.20462, ("lbs", "kg"): 0.453592,
        ("g", "oz"): 0.035274, ("oz", "g"): 28.3495,
        # Volume
        ("liters", "gallons"): 0.264172, ("gallons", "liters"): 3.78541,
        ("ml", "oz"): 0.033814, ("oz", "ml"): 29.5735,
        # Speed
        ("km/h", "mph"): 0.621371, ("mph", "km/h"): 1.60934,
    }

    f, t = from_unit.lower(), to_unit.lower()

    # Temperature special cases
    if f in ("celsius", "c") and t in ("fahrenheit", "f"):
        return f"{round(value * 9/5 + 32, 2)} {to_unit}"
    if f in ("fahrenheit", "f") and t in ("celsius", "c"):
        return f"{round((value - 32) * 5/9, 2)} {to_unit}"

    factor = conversions.get((f, t))
    if factor is None:
        return f"Unknown conversion: {from_unit} to {to_unit}"
    return f"{round(value * factor, 4)} {to_unit}"


BUILTIN_TOOLS = [
    get_current_time,
    set_timer,
    check_timer,
    cancel_timer,
    calculate,
    roll_dice,
    random_number,
    flip_coin,
    unit_convert,
]
