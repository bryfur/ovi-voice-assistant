"""Tests for the built-in agent tools module."""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock

import pytest

from ovi_voice_assistant.agent.assistant_context import AssistantContext
from ovi_voice_assistant.agent.tools import (
    calculate,
    cancel_timer,
    check_timer,
    flip_coin,
    get_current_time,
    now_playing,
    pause_music,
    random_number,
    resume_music,
    roll_dice,
    say,
    set_timer,
    skip_track,
    stop_music,
    unit_convert,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_ctx(name: str) -> MagicMock:
    """Create a mock ToolContext with the given tool name (no AssistantContext)."""
    ctx = MagicMock()
    ctx.tool_name = name
    return ctx


def _tool_ctx_with_assistant(name: str, assistant: AssistantContext) -> MagicMock:
    """Create a mock ToolContext whose .context is a real AssistantContext."""
    ctx = MagicMock()
    ctx.tool_name = name
    ctx.context = assistant
    return ctx


async def _invoke(
    tool, params: dict | None = None, ctx: MagicMock | None = None
) -> str:
    """Invoke a FunctionTool's on_invoke_tool with a mock context."""
    if ctx is None:
        ctx = _tool_ctx(tool.name)
    return await tool.on_invoke_tool(ctx, json.dumps(params or {}))


# ---------------------------------------------------------------------------
# say
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSay:
    async def test_calls_say_callback(self) -> None:
        say_cb = AsyncMock()
        assistant_ctx = AssistantContext(say=say_cb)
        ctx = _tool_ctx_with_assistant("say", assistant_ctx)

        result = await _invoke(say, {"text": "Hold on"}, ctx=ctx)

        say_cb.assert_awaited_once_with("Hold on")
        assert result == "Spoken."

    async def test_no_callback(self) -> None:
        assistant_ctx = AssistantContext()
        ctx = _tool_ctx_with_assistant("say", assistant_ctx)

        result = await _invoke(say, {"text": "Hello"}, ctx=ctx)

        assert "not available" in result.lower()


# ---------------------------------------------------------------------------
# get_current_time
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGetCurrentTime:
    async def test_returns_formatted_string(self) -> None:
        result = await _invoke(get_current_time)

        # Should contain day-of-week and year
        assert "202" in result  # year
        assert "at" in result  # "at HH:MM"

    async def test_with_utc_timezone(self) -> None:
        result = await _invoke(get_current_time, {"timezone": "UTC"})

        assert "UTC" in result

    async def test_with_named_timezone(self) -> None:
        result = await _invoke(get_current_time, {"timezone": "America/New_York"})

        # Should contain a timezone abbreviation (EDT or EST)
        assert any(tz in result for tz in ("EDT", "EST"))

    async def test_invalid_timezone_returns_error(self) -> None:
        result = await _invoke(get_current_time, {"timezone": "Not/A/Zone"})

        assert "error" in result.lower() or "Error" in result


# ---------------------------------------------------------------------------
# calculate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCalculate:
    async def test_basic_addition(self) -> None:
        result = await _invoke(calculate, {"expression": "2 + 3"})

        assert result == "5"

    async def test_basic_multiplication(self) -> None:
        result = await _invoke(calculate, {"expression": "6 * 7"})

        assert result == "42"

    async def test_division_returns_float(self) -> None:
        result = await _invoke(calculate, {"expression": "7 / 2"})

        assert result == "3.5"

    async def test_integer_float_coerced(self) -> None:
        result = await _invoke(calculate, {"expression": "2.0 * 2"})

        # 4.0 should be returned as "4"
        assert result == "4"

    async def test_exponentiation(self) -> None:
        result = await _invoke(calculate, {"expression": "2 ** 10"})

        assert result == "1024"

    async def test_sqrt(self) -> None:
        result = await _invoke(calculate, {"expression": "sqrt(144)"})

        assert result == "12"

    async def test_pi(self) -> None:
        result = await _invoke(calculate, {"expression": "pi"})

        assert float(result) == pytest.approx(math.pi)

    async def test_e(self) -> None:
        result = await _invoke(calculate, {"expression": "e"})

        assert float(result) == pytest.approx(math.e)

    async def test_sin_pi_half(self) -> None:
        result = await _invoke(calculate, {"expression": "sin(pi / 2)"})

        assert float(result) == pytest.approx(1.0)

    async def test_cos_zero(self) -> None:
        result = await _invoke(calculate, {"expression": "cos(0)"})

        assert float(result) == pytest.approx(1.0)

    async def test_log(self) -> None:
        result = await _invoke(calculate, {"expression": "log(e)"})

        assert float(result) == pytest.approx(1.0)

    async def test_log10(self) -> None:
        result = await _invoke(calculate, {"expression": "log10(1000)"})

        assert float(result) == pytest.approx(3.0)

    async def test_ceil(self) -> None:
        result = await _invoke(calculate, {"expression": "ceil(2.3)"})

        assert result == "3"

    async def test_floor(self) -> None:
        result = await _invoke(calculate, {"expression": "floor(2.9)"})

        assert result == "2"

    async def test_abs(self) -> None:
        result = await _invoke(calculate, {"expression": "abs(-5)"})

        assert result == "5"

    async def test_round(self) -> None:
        result = await _invoke(calculate, {"expression": "round(3.14159, 2)"})

        assert result == "3.14"

    async def test_invalid_expression_returns_error(self) -> None:
        result = await _invoke(calculate, {"expression": "definitely not math"})

        assert result.startswith("Error:")

    async def test_builtins_blocked(self) -> None:
        result = await _invoke(calculate, {"expression": "__import__('os')"})

        # Should not allow access to __import__ or other builtins
        assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# roll_dice
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRollDice:
    async def test_single_d6(self) -> None:
        result = await _invoke(roll_dice, {"sides": 6, "count": 1})

        value = int(result)
        assert 1 <= value <= 6

    async def test_defaults(self) -> None:
        result = await _invoke(roll_dice, {})

        # defaults: sides=6, count=1
        value = int(result)
        assert 1 <= value <= 6

    async def test_multiple_dice(self) -> None:
        result = await _invoke(roll_dice, {"sides": 6, "count": 3})

        assert "total:" in result
        # Parse the list from the string "[a, b, c] (total: N)"
        list_part = result.split("(")[0].strip()
        values = json.loads(list_part)
        assert len(values) == 3
        assert all(1 <= v <= 6 for v in values)

    async def test_d20(self) -> None:
        result = await _invoke(roll_dice, {"sides": 20, "count": 1})

        value = int(result)
        assert 1 <= value <= 20

    async def test_multiple_dice_total_correct(self) -> None:
        result = await _invoke(roll_dice, {"sides": 6, "count": 4})

        list_part = result.split("(")[0].strip()
        total_part = result.split("total: ")[1].rstrip(")")
        values = json.loads(list_part)
        assert sum(values) == int(total_part)


# ---------------------------------------------------------------------------
# random_number
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRandomNumber:
    async def test_default_range(self) -> None:
        result = await _invoke(random_number, {})

        value = int(result)
        assert 1 <= value <= 100

    async def test_custom_range(self) -> None:
        result = await _invoke(random_number, {"low": 50, "high": 60})

        value = int(result)
        assert 50 <= value <= 60

    async def test_single_value_range(self) -> None:
        result = await _invoke(random_number, {"low": 7, "high": 7})

        assert result == "7"


# ---------------------------------------------------------------------------
# flip_coin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFlipCoin:
    async def test_returns_heads_or_tails(self) -> None:
        result = await _invoke(flip_coin)

        assert result in ("Heads", "Tails")

    async def test_multiple_flips_produce_both_outcomes(self) -> None:
        results = set()
        for _ in range(50):
            results.add(await _invoke(flip_coin))

        # With 50 flips, extremely unlikely to get only one outcome
        assert results == {"Heads", "Tails"}


# ---------------------------------------------------------------------------
# unit_convert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnitConvert:
    # Length
    async def test_km_to_miles(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 10, "from_unit": "km", "to_unit": "miles"}
        )

        assert "6.2137" in result
        assert "miles" in result

    async def test_miles_to_km(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "miles", "to_unit": "km"}
        )

        assert "1.6093" in result

    async def test_m_to_ft(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "m", "to_unit": "ft"}
        )

        assert "3.2808" in result

    async def test_ft_to_m(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 10, "from_unit": "ft", "to_unit": "m"}
        )

        assert "3.048" in result

    async def test_cm_to_in(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 100, "from_unit": "cm", "to_unit": "in"}
        )

        assert "39.3701" in result

    async def test_in_to_cm(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "in", "to_unit": "cm"}
        )

        assert "2.54" in result

    # Weight
    async def test_kg_to_lbs(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "kg", "to_unit": "lbs"}
        )

        assert "2.2046" in result

    async def test_lbs_to_kg(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "lbs", "to_unit": "kg"}
        )

        assert "0.4536" in result

    # Volume
    async def test_liters_to_gallons(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "liters", "to_unit": "gallons"}
        )

        assert "0.2642" in result

    async def test_gallons_to_liters(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "gallons", "to_unit": "liters"}
        )

        assert "3.7854" in result

    # Speed
    async def test_kmh_to_mph(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 100, "from_unit": "km/h", "to_unit": "mph"}
        )

        assert "62.1371" in result

    async def test_mph_to_kmh(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 60, "from_unit": "mph", "to_unit": "km/h"}
        )

        assert "96.5604" in result

    # Temperature
    async def test_celsius_to_fahrenheit(self) -> None:
        result = await _invoke(
            unit_convert,
            {"value": 100, "from_unit": "celsius", "to_unit": "fahrenheit"},
        )

        assert "212" in result

    async def test_fahrenheit_to_celsius(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 32, "from_unit": "fahrenheit", "to_unit": "celsius"}
        )

        assert "0" in result

    async def test_celsius_short_form(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 0, "from_unit": "C", "to_unit": "F"}
        )

        assert "32" in result

    async def test_fahrenheit_short_form(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 212, "from_unit": "F", "to_unit": "C"}
        )

        assert "100" in result

    async def test_case_insensitive(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "KM", "to_unit": "Miles"}
        )

        assert "0.6214" in result

    # Unknown
    async def test_unknown_conversion(self) -> None:
        result = await _invoke(
            unit_convert, {"value": 1, "from_unit": "parsecs", "to_unit": "lightyears"}
        )

        assert "Unknown conversion" in result
        assert "parsecs" in result


# ---------------------------------------------------------------------------
# set_timer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSetTimer:
    @pytest.fixture
    def assistant_ctx(self) -> AssistantContext:
        return AssistantContext()

    async def test_zero_duration_rejected(
        self, assistant_ctx: AssistantContext
    ) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 0, "seconds": 0}, ctx=ctx)

        assert "must be greater than zero" in result

    async def test_negative_duration_rejected(
        self, assistant_ctx: AssistantContext
    ) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": -1}, ctx=ctx)

        assert "must be greater than zero" in result

    async def test_seconds_only(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"seconds": 30}, ctx=ctx)

        assert "30 seconds" in result
        assert "timer" in result.lower()
        assistant_ctx.cancel_timer("timer")

    async def test_minutes_only(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 5}, ctx=ctx)

        assert "5 minutes" in result
        assistant_ctx.cancel_timer("timer")

    async def test_minutes_and_seconds(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 2, "seconds": 30}, ctx=ctx)

        assert "2 minutes" in result
        assert "30 seconds" in result
        assistant_ctx.cancel_timer("timer")

    async def test_hours_formatting(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 90}, ctx=ctx)

        assert "1 hour" in result
        assert "30 minutes" in result
        assistant_ctx.cancel_timer("timer")

    async def test_plural_hours(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 120}, ctx=ctx)

        assert "2 hours" in result
        assistant_ctx.cancel_timer("timer")

    async def test_singular_units(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 1, "seconds": 1}, ctx=ctx)

        assert "1 minute" in result
        assert "1 second" in result
        # Verify no spurious 's'
        assert "1 minutes" not in result
        assert "1 seconds" not in result
        assistant_ctx.cancel_timer("timer")

    async def test_custom_label(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        result = await _invoke(set_timer, {"minutes": 5, "label": "pasta"}, ctx=ctx)

        assert "'pasta'" in result
        assistant_ctx.cancel_timer("pasta")

    async def test_timer_actually_scheduled(
        self, assistant_ctx: AssistantContext
    ) -> None:
        ctx = _tool_ctx_with_assistant("set_timer", assistant_ctx)

        await _invoke(set_timer, {"seconds": 60, "label": "test"}, ctx=ctx)

        assert "test" in assistant_ctx._timers
        assistant_ctx.cancel_timer("test")


# ---------------------------------------------------------------------------
# check_timer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckTimer:
    @pytest.fixture
    def assistant_ctx(self) -> AssistantContext:
        return AssistantContext()

    async def test_no_active_timers(self, assistant_ctx: AssistantContext) -> None:
        ctx = _tool_ctx_with_assistant("check_timer", assistant_ctx)

        result = await _invoke(check_timer, ctx=ctx)

        assert result == "No active timers."

    async def test_with_active_timer(self, assistant_ctx: AssistantContext) -> None:
        assistant_ctx.schedule_timer(300, "eggs")
        ctx = _tool_ctx_with_assistant("check_timer", assistant_ctx)

        result = await _invoke(check_timer, ctx=ctx)

        assert "'eggs'" in result
        assert "remaining" in result
        assistant_ctx.cancel_timer("eggs")

    async def test_multiple_timers(self, assistant_ctx: AssistantContext) -> None:
        assistant_ctx.schedule_timer(60, "tea")
        assistant_ctx.schedule_timer(600, "laundry")
        ctx = _tool_ctx_with_assistant("check_timer", assistant_ctx)

        result = await _invoke(check_timer, ctx=ctx)

        assert "'tea'" in result
        assert "'laundry'" in result
        assert ";" in result  # multiple timers separated by semicolon
        assistant_ctx.cancel_timer("tea")
        assistant_ctx.cancel_timer("laundry")

    async def test_timer_format_includes_time_units(
        self, assistant_ctx: AssistantContext
    ) -> None:
        assistant_ctx.schedule_timer(3661, "long")
        ctx = _tool_ctx_with_assistant("check_timer", assistant_ctx)

        result = await _invoke(check_timer, ctx=ctx)

        # Should have hours, minutes, and seconds
        assert "h" in result
        assert "m" in result
        assert "s" in result
        assistant_ctx.cancel_timer("long")


# ---------------------------------------------------------------------------
# cancel_timer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCancelTimer:
    @pytest.fixture
    def assistant_ctx(self) -> AssistantContext:
        return AssistantContext()

    async def test_cancel_existing_timer(self, assistant_ctx: AssistantContext) -> None:
        assistant_ctx.schedule_timer(60, "timer")
        ctx = _tool_ctx_with_assistant("cancel_timer", assistant_ctx)

        result = await _invoke(cancel_timer, {"label": "timer"}, ctx=ctx)

        assert "cancelled" in result
        assert "'timer'" in result

    async def test_cancel_nonexistent_timer(
        self, assistant_ctx: AssistantContext
    ) -> None:
        ctx = _tool_ctx_with_assistant("cancel_timer", assistant_ctx)

        result = await _invoke(cancel_timer, {"label": "nope"}, ctx=ctx)

        assert "No active timer named 'nope'" in result

    async def test_cancel_default_label(self, assistant_ctx: AssistantContext) -> None:
        assistant_ctx.schedule_timer(60, "timer")
        ctx = _tool_ctx_with_assistant("cancel_timer", assistant_ctx)

        # Default label is "timer"
        result = await _invoke(cancel_timer, {}, ctx=ctx)

        assert "cancelled" in result

    async def test_cancel_removes_from_timers(
        self, assistant_ctx: AssistantContext
    ) -> None:
        assistant_ctx.schedule_timer(60, "toast")
        ctx = _tool_ctx_with_assistant("cancel_timer", assistant_ctx)

        await _invoke(cancel_timer, {"label": "toast"}, ctx=ctx)

        assert "toast" not in assistant_ctx._timers


# ---------------------------------------------------------------------------
# Music tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPauseMusic:
    async def test_no_player(self) -> None:
        assistant = AssistantContext(music_player=None)
        ctx = _tool_ctx_with_assistant("pause_music", assistant)

        result = await _invoke(pause_music, ctx=ctx)

        assert "No music is playing" in result

    async def test_empty_queue(self) -> None:
        player = MagicMock()
        player.queue = []
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("pause_music", assistant)

        result = await _invoke(pause_music, ctx=ctx)

        assert "No music is playing" in result

    async def test_pauses_current_track(self) -> None:
        track = MagicMock()
        track.title = "Song"
        track.artist = "Artist"
        player = MagicMock()
        player.queue = [track]
        player.get_current.return_value = track
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("pause_music", assistant)

        result = await _invoke(pause_music, ctx=ctx)

        player.pause.assert_called_once()
        assert "Song" in result
        assert "Artist" in result


@pytest.mark.asyncio
class TestResumeMusic:
    async def test_no_music_to_resume(self) -> None:
        assistant = AssistantContext(music_player=None)
        ctx = _tool_ctx_with_assistant("resume_music", assistant)

        result = await _invoke(resume_music, ctx=ctx)

        assert "No music to resume" in result

    async def test_resumes_track(self) -> None:
        track = MagicMock()
        track.title = "Song"
        track.artist = "Artist"
        player = MagicMock()
        player.queue = [track]
        player.get_current.return_value = track
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("resume_music", assistant)

        result = await _invoke(resume_music, ctx=ctx)

        player.resume.assert_called_once()
        assert "Resuming" in result
        assert "Song" in result


@pytest.mark.asyncio
class TestSkipTrack:
    async def test_no_music_playing(self) -> None:
        assistant = AssistantContext(music_player=None)
        ctx = _tool_ctx_with_assistant("skip_track", assistant)

        result = await _invoke(skip_track, ctx=ctx)

        assert "No music is playing" in result

    async def test_skip_to_next(self) -> None:
        track = MagicMock()
        track.title = "Next"
        track.artist = "Band"
        player = MagicMock()
        player.queue = [MagicMock(), track]
        player.skip.return_value = track
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("skip_track", assistant)

        result = await _invoke(skip_track, ctx=ctx)

        assert "Next" in result
        assert "Band" in result

    async def test_skip_end_of_queue(self) -> None:
        player = MagicMock()
        player.queue = [MagicMock()]
        player.skip.return_value = None
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("skip_track", assistant)

        result = await _invoke(skip_track, ctx=ctx)

        assert "No more tracks" in result


@pytest.mark.asyncio
class TestStopMusic:
    async def test_no_music(self) -> None:
        assistant = AssistantContext(music_player=None)
        ctx = _tool_ctx_with_assistant("stop_music", assistant)

        result = await _invoke(stop_music, ctx=ctx)

        assert "No music is playing" in result

    async def test_stops_playback(self) -> None:
        player = MagicMock()
        player.queue = [MagicMock()]
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("stop_music", assistant)

        result = await _invoke(stop_music, ctx=ctx)

        player.stop.assert_called_once()
        assert "stopped" in result.lower()


@pytest.mark.asyncio
class TestNowPlaying:
    async def test_no_player(self) -> None:
        assistant = AssistantContext(music_player=None)
        ctx = _tool_ctx_with_assistant("now_playing", assistant)

        result = await _invoke(now_playing, ctx=ctx)

        assert "No music is playing" in result

    async def test_nothing_playing(self) -> None:
        player = MagicMock()
        player.queue = [MagicMock()]
        player.get_current.return_value = None
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("now_playing", assistant)

        result = await _invoke(now_playing, ctx=ctx)

        assert "No music is playing" in result

    async def test_shows_current_track(self) -> None:
        track = MagicMock()
        track.title = "Bohemian Rhapsody"
        track.artist = "Queen"
        player = MagicMock()
        player.queue = [track, MagicMock()]
        player.get_current.return_value = track
        player.current_index = 0
        player.is_active = True
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("now_playing", assistant)

        result = await _invoke(now_playing, ctx=ctx)

        assert "Bohemian Rhapsody" in result
        assert "Queen" in result
        assert "playing" in result
        assert "track 1 of 2" in result

    async def test_shows_paused_status(self) -> None:
        track = MagicMock()
        track.title = "Song"
        track.artist = "Artist"
        player = MagicMock()
        player.queue = [track]
        player.get_current.return_value = track
        player.current_index = 0
        player.is_active = False
        assistant = AssistantContext(music_player=player)
        ctx = _tool_ctx_with_assistant("now_playing", assistant)

        result = await _invoke(now_playing, ctx=ctx)

        assert "paused" in result


# ---------------------------------------------------------------------------
# BUILTIN_TOOLS list
# ---------------------------------------------------------------------------


class TestBuiltinToolsList:
    def test_all_tools_present(self) -> None:
        from ovi_voice_assistant.agent.tools import BUILTIN_TOOLS

        names = {t.name for t in BUILTIN_TOOLS}

        expected = {
            "say",
            "get_current_time",
            "set_timer",
            "check_timer",
            "cancel_timer",
            "calculate",
            "roll_dice",
            "random_number",
            "flip_coin",
            "unit_convert",
            "play_music",
            "pause_music",
            "resume_music",
            "skip_track",
            "stop_music",
            "now_playing",
            "create_automation",
            "list_automations",
            "delete_automation",
            "toggle_automation",
        }
        assert names == expected
