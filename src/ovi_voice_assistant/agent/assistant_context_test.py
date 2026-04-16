"""Tests for AssistantContext class."""

from __future__ import annotations

import asyncio

import pytest

from ovi_voice_assistant.agent.assistant_context import AssistantContext


@pytest.fixture
def ctx() -> AssistantContext:
    return AssistantContext()


@pytest.mark.asyncio
async def test_schedule_timer_adds_to_timers(ctx: AssistantContext) -> None:
    ctx.schedule_timer(10.0, "eggs")

    assert "eggs" in ctx._timers
    handle, fire_time = ctx._timers["eggs"]
    assert fire_time > 0
    handle.cancel()


@pytest.mark.asyncio
async def test_cancel_timer_returns_true_for_existing(ctx: AssistantContext) -> None:
    ctx.schedule_timer(10.0, "tea")

    result = ctx.cancel_timer("tea")

    assert result is True
    assert "tea" not in ctx._timers


@pytest.mark.asyncio
async def test_cancel_timer_returns_false_for_nonexistent(
    ctx: AssistantContext,
) -> None:
    result = ctx.cancel_timer("nonexistent")

    assert result is False


@pytest.mark.asyncio
async def test_get_timer_status_returns_remaining_seconds(
    ctx: AssistantContext,
) -> None:
    ctx.schedule_timer(30.0, "pasta")

    status = ctx.get_timer_status()

    assert "pasta" in status
    # Should be close to 30 seconds remaining (within 1s tolerance)
    assert 29.0 <= status["pasta"] <= 30.0
    ctx.cancel_timer("pasta")


@pytest.mark.asyncio
async def test_timer_fire_calls_announce(ctx: AssistantContext) -> None:
    called_with: list[str] = []

    def fake_announce(text: str) -> asyncio.Task:
        called_with.append(text)
        return asyncio.ensure_future(asyncio.sleep(0))

    ctx.announce = fake_announce
    ctx.schedule_timer(0.01, "coffee")

    # Let the timer fire
    await asyncio.sleep(0.05)

    assert len(called_with) == 1
    assert "coffee" in called_with[0]


@pytest.mark.asyncio
async def test_timer_fire_removes_from_timers(ctx: AssistantContext) -> None:
    ctx.announce = lambda text: asyncio.ensure_future(asyncio.sleep(0))
    ctx.schedule_timer(0.01, "toast")

    await asyncio.sleep(0.05)

    assert "toast" not in ctx._timers


@pytest.mark.asyncio
async def test_timer_fire_without_announce(ctx: AssistantContext) -> None:
    """Timer fires with no announce callback set -- should not raise."""
    ctx.announce = None
    ctx.schedule_timer(0.01, "silent")

    await asyncio.sleep(0.05)

    # Timer should still be cleaned up from _timers
    assert "silent" not in ctx._timers


@pytest.mark.asyncio
async def test_multiple_timers(ctx: AssistantContext) -> None:
    ctx.schedule_timer(10.0, "alpha")
    ctx.schedule_timer(20.0, "beta")
    ctx.schedule_timer(30.0, "gamma")

    status = ctx.get_timer_status()

    assert len(ctx._timers) == 3
    assert status["alpha"] < status["beta"] < status["gamma"]

    # Cleanup
    for label in ["alpha", "beta", "gamma"]:
        ctx.cancel_timer(label)


@pytest.mark.asyncio
async def test_cancel_then_status_shows_removed(ctx: AssistantContext) -> None:
    ctx.schedule_timer(10.0, "nap")
    ctx.schedule_timer(20.0, "walk")
    ctx.cancel_timer("nap")

    status = ctx.get_timer_status()

    assert "nap" not in status
    assert "walk" in status

    ctx.cancel_timer("walk")
